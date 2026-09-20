#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
ARKODE (SUNDIALS) as an Assimulo solver: adaptive one-step Runge-Kutta methods,
implicit (DIRK/ESDIRK) or explicit, with ARKODE's rootfinding for state events.
Requires SUNDIALS >= 7.1 (the stepper-independent ARKode* API).
"""

import numpy as np
cimport numpy as np

from assimulo.exception import AssimuloException
from assimulo.explicit_ode cimport Explicit_ODE
from assimulo.support import set_type_shape_array

cimport sundials_includes as SUNDIALS
cimport arkode_includes as ARK
from arkode_includes cimport (ARK_NORMAL, ARK_ONE_STEP, ARK_SUCCESS, ARK_TSTOP_RETURN, ARK_ROOT_RETURN,
                              ARK_TOO_MUCH_WORK, ARK_TOO_MUCH_ACC, ARK_ERR_FAILURE, ARK_CONV_FAILURE,
                              ARK_LINIT_FAIL, ARK_LSETUP_FAIL, ARK_LSOLVE_FAIL, ARK_RHSFUNC_FAIL,
                              ARK_FIRST_RHSFUNC_ERR, ARK_REPTD_RHSFUNC_ERR, ARK_UNREC_RHSFUNC_ERR,
                              ARK_RTFUNC_FAIL, ARK_MEM_FAIL, ARK_MEM_NULL, ARK_ILL_INPUT, ARK_TOO_CLOSE,
                              ARK_INTERP_FAIL, ARK_STEPPER_UNSUPPORTED)

from sundials_includes cimport N_Vector, realtype, N_VectorContent_Serial, DENSE_COL, sunindextype
from sundials_includes cimport memcpy, N_VNew_Serial, DlsMat, SUNMatrix, SUNMatrixContent_Dense, SUNMatrixContent_Sparse
from sundials_includes cimport malloc, free, N_VConst_Serial
from sundials_includes cimport N_VCloneVectorArray, N_VDestroy

include "constants.pxi"                          # Assimulo's ID_* flags
include "../lib/sundials_constants.pxi"
include "../lib/sundials_callbacks.pxi"          # arr2nv, nv2arr
include "../lib/sundials_callbacks_ida_cvode.pxi" # cv_rhs, cv_jac, cv_jacv, cv_root, cv_err, ProblemData



cdef class ARKODE(Explicit_ODE):
    r"""
    SUNDIALS ARKODE: adaptive one-step Runge-Kutta methods for

    .. math::

        \dot{y} = f(t,y), \quad y(t_0) = y_0.

    ``method = "implicit"`` (default) runs ARKStep with a diagonally implicit RK
    table (orders 2-5, L-stable ones at every order), Newton iteration and a direct
    dense linear solver with the problem's Jacobian if ``usejac``;
    ``method = "explicit"`` runs an explicit RK table (orders 2-9) without any
    linear algebra. The table is chosen by ``order`` (ARKODE's default table of that
    order) or by name with ``table`` (e.g. ``"ARKODE_TRBDF2_3_3_2"``,
    ``"ARKODE_ESDIRK324L2SA_4_2_3"``, ``"ARKODE_DORMAND_PRINCE_7_4_5"``).

    Being one-step methods, they restart at full order after every event; state
    events are located by ARKODE's rootfinding (``external_event_detection = False``)
    or by Assimulo's locator on the dense output, time events are stop times.
    Only the stepper-independent ARKode* API of SUNDIALS >= 7.1 is used.

    On a rhs with a jump the solution rides on (a switch without an event indicator)
    Newton fails on most stages of a many-stage method whatever the Jacobian, while a
    2-stage method gets through: when more than ``fallback_conv_fail_rate`` of the last
    ``fallback_window`` step attempts failed by nonlinear convergence, the run continues
    with ``fallback_table`` (default TRBDF2) from that point, once per simulation, logged
    at normal verbosity and shown in the statistics; ``fallback_time`` records when.
    """
    cdef void* ark_mem
    cdef ProblemData pData
    cdef N_Vector yTemp, nv_atol
    cdef SUNDIALS.SUNContext ctx
    cdef SUNDIALS.SUNMatrix sun_matrix
    cdef SUNDIALS.SUNLinearSolver sun_linearsolver
    cdef object pt_root, pt_fcn, pt_jac, pt_jacv
    cdef public object event_func
    cdef public np.ndarray g_old
    cdef object _created_method     # the 'method' the memory block was created for
    cdef dict _last_counters        # ARKODE's cumulative counters at the last store_statistics
    cdef int _fresh_memory          # the memory block was just created: the method-defining options are still to be set
    cdef object _active_table       # the table in use: options['table'] or, after a fallback, options['fallback_table']
    cdef long int _fb_attempts0, _fb_fails0, _fb_rhsfails0   # counters at the start of the current failure-rate window
    cdef public object fallback_time  # time of the fallback to the 2-stage table, None if it did not happen
    cdef public object fallback_return_time   # time of the switch back to the user's table after a hard failure of the fallback

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        self.pData = ProblemData()
        self.ark_mem = NULL
        self.sun_matrix = NULL
        self.sun_linearsolver = NULL
        self.yTemp = NULL
        self.nv_atol = NULL
        self.ctx = NULL
        self._created_method = None
        self._last_counters = {}
        self._fresh_memory = 0
        self._active_table = None
        self._fb_attempts0 = 0
        self._fb_fails0 = 0
        self._fb_rhsfails0 = 0
        self.fallback_time = None
        self.fallback_return_time = None
        SUNDIALS.SUNContext_Create(SUNDIALS.SUN_COMM_NULL, &self.ctx)

        self.set_problem_data()

        # Solver options (names shared with CVode where the meaning is the same)
        self.options["atol"] = np.array([1.0e-6] * self.problem_info["dim"])
        self.options["rtol"] = 1.0e-6
        self.options["maxh"] = 0.0            # 0: no maximum step
        self.options["minh"] = 0.0
        self.options["inith"] = 0.0           # 0: ARKODE estimates the first step
        self.options["maxsteps"] = 10000
        self.options["usejac"] = True if (self.problem_info["jac_fcn"] or self.problem_info["jacv_fcn"]) else False
        self.options["linear_solver"] = "DENSE"   # or SPGMR (Jacobian-vector products)
        self.options["maxkrylov"] = 5
        self.options["external_event_detection"] = False   # ARKODE rootfinding by default
        # ARKODE-specific
        self.options["method"] = "implicit"   # "implicit" (DIRK) or "explicit" (ERK)
        self.options["order"] = 4             # method order when 'table' is None
        self.options["table"] = None          # Butcher table name, overrides 'order'
        self.options["predictor"] = 2         # ARKODE predictor 0-5; 0 (y_n) needs 2.4x the steps on Van der Pol, 2 (variable-order) halves the Newton failures of 1 on the data-center FMU
        self.options["nonlin_conv_coef"] = 0.1
        self.options["max_nonlin_iters"] = 4  # ARKODE default 3; 4 saves 60 % of the Jacobians on Van der Pol
        self.options["deduce_implicit_rhs"] = True   # stage derivatives from the stage equations
        self.options["interpolant_degree"] = 3       # -1: ARKODE's default for the method (its order); 3: the cubic Hermite interpolant -- the quintic of order 5 costs rhs evaluations at every output point and reports spurious events under rootfinding, orders <= 4 are unchanged
        self.options["maxncf"] = 10           # max convergence failures per step
        self.options["lsetup_frequency"] = 20     # steps between linear-solver setups (ARKODE default 20)
        self.options["jac_eval_frequency"] = 51   # setups between Jacobian evaluations (ARKODE default 51)
        self.options["delta_gamma_max"] = 0.05    # relative change of gamma that forces a new setup (ARKODE default 0.2; 0.05 saves 10-15 % rhs and most error-test failures on the FMUs)
        self.options["maxnef"] = 7            # max error test failures per step
        self.options["restart_h"] = "estimate"   # after an event: "estimate" a new first step or "keep" the last
        # step-size adaptivity bounds (ARKODE defaults: safety 0.96, max_growth 20, max_first_growth 1e4,
        # max_efail_growth 0.3, max_cfail_growth 0.25)
        self.options["safety"] = 0.96
        self.options["max_growth"] = 20.0
        self.options["max_first_growth"] = 10000.0
        self.options["max_efail_growth"] = 0.3
        self.options["max_cfail_growth"] = 0.25
        self.options["report_continuously"] = False
        # Fallback for a discontinuous rhs (a sliding mode, e.g. an all-or-nothing phase change
        # without an event indicator): Newton on the stages of a many-stage method fails on most
        # steps there whatever the Jacobian, while a 2-stage method rides it. When more than
        # 'fallback_conv_fail_rate' of the last 'fallback_window' step attempts failed by nonlinear
        # convergence (steps failed by a recoverable rhs failure not counted: those are the
        # model's, not the method's), the memory is recreated with 'fallback_table' at the current point and the
        # run continues with it (once per simulation, logged; None disables). If the fallback
        # table then fails hard (repeated error-test or convergence failures), the run switches
        # back to the user's table at that point, once.
        self.options["fallback_table"] = "ARKODE_TRBDF2_3_3_2"
        self.options["fallback_conv_fail_rate"] = 0.25
        self.options["fallback_window"] = 50

        self.statistics.add_key("nstepattempts", "Number of step attempts")
        self.statistics.add_key("nconvfails", "Number of steps failed by nonlinear convergence")

        self.supports["report_continuously"] = True
        self.supports["interpolated_output"] = True
        self.supports["state_events"] = True

    def __dealloc__(self):
        if self.yTemp != NULL:
            N_VDestroy(self.yTemp)
        if self.nv_atol != NULL:
            N_VDestroy(self.nv_atol)
        if self.ark_mem != NULL:
            ARK.ARKodeFree(&self.ark_mem)
        if self.sun_matrix != NULL:
            SUNDIALS.SUNMatDestroy(self.sun_matrix)
        if self.sun_linearsolver != NULL:
            SUNDIALS.SUNLinSolFree(self.sun_linearsolver)
        if self.ctx != NULL:
            ARK.SUNContext_Free(&self.ctx)

    # ------------------------------------------------------------------ set-up
    cdef set_problem_data(self):
        self.pt_fcn = self.problem.rhs
        self.pData.RHS = <void*>self.pt_fcn
        self.pData.dim = self.problem_info["dim"]
        self.pData.memSize = self.pData.dim * sizeof(realtype)
        if self.problem_info["state_events"] is True:
            self.pt_root = self.problem.state_events
            self.pData.ROOT = <void*>self.pt_root
            self.pData.dimRoot = self.problem_info["dimRoot"]
            self.pData.memSizeRoot = self.pData.dimRoot * sizeof(realtype)
        if self.problem_info["jac_fcn"] is True:
            self.pt_jac = self.problem.jac
            self.pData.JAC = <void*>self.pt_jac
            self.pData.memSizeJac = self.pData.dim * self.pData.dim * sizeof(realtype)
        if self.problem_info["jacv_fcn"] is True:
            self.pt_jacv = self.problem.jacv
            self.pData.JACV = <void*>self.pt_jacv
        self.pData.verbose = 2
        self.pData.create_work_arrays()

    cdef N_Vector _arr2nv(self, x):
        """A serial N_Vector holding a copy of `x`, created in this solver's SUNContext
        (the generic arr2nv of sundials_callbacks.pxi creates and leaks a new context per
        call; each context also reopens the SUNLogger files from the environment, which
        truncates a log written for debugging)."""
        cdef np.ndarray[realtype, ndim=1, mode='c'] ndx = np.array(x, dtype=np.float64)
        cdef N_Vector v = N_VNew_Serial(len(ndx), self.ctx)
        memcpy((<N_VectorContent_Serial>v.content).data, <void*>PyArray_DATA(ndx), len(ndx)*sizeof(realtype))
        return v

    cpdef initialize(self):
        self.statistics.reset()
        if self._active_table != self.options["table"] and self.ark_mem != NULL:
            self._free_memory()          # a previous simulation fell back: start again from the user's table
        self._active_table = self.options["table"]
        self.fallback_time = None
        self.fallback_return_time = None
        self._fb_attempts0 = 0
        self._fb_fails0 = 0
        self._fb_rhsfails0 = 0
        self.pData.nrhsfails = 0
        self.initialize_arkode()

    cdef _free_memory(self):
        if self.ark_mem != NULL:
            ARK.ARKodeFree(&self.ark_mem)
            self.ark_mem = NULL
        self._last_counters = {}
        self._free_linear_solver()

    cdef _free_linear_solver(self):
        if self.sun_linearsolver != NULL:
            SUNDIALS.SUNLinSolFree(self.sun_linearsolver)
            self.sun_linearsolver = NULL
        if self.sun_matrix != NULL:
            SUNDIALS.SUNMatDestroy(self.sun_matrix)
            self.sun_matrix = NULL

    cdef initialize_arkode(self):
        """Creates the ARKODE memory on the first call, resets it to (t, y) afterwards."""
        cdef int flag
        method = self.options["method"]
        if method not in ("implicit", "explicit"):
            raise ARKODEError(ARK_ILL_INPUT, self.t, "'method' must be 'implicit' or 'explicit'")

        if self.yTemp != NULL:
            N_VDestroy(self.yTemp)
        self.yTemp = self._arr2nv(self.y)
        self.pData.verbose = 2 if self.verbosity <= NORMAL else 3

        if self.problem_info["switches"]:
            self.pData.sw = <void*>self.sw

        if self.ark_mem != NULL and self._created_method != method:
            self._free_memory()

        if self.ark_mem == NULL:
            if method == "implicit":
                self.ark_mem = ARK.ARKStepCreate(NULL, cv_rhs, self.t, self.yTemp, self.ctx)
            else:
                self.ark_mem = ARK.ARKStepCreate(cv_rhs, NULL, self.t, self.yTemp, self.ctx)
            if self.ark_mem == NULL:
                raise ARKODEError(ARK_MEM_FAIL, self.t)
            self._created_method = method
            self._fresh_memory = 1

            if self.problem_info["state_events"]:
                if self.options["external_event_detection"]:
                    flag = ARK.ARKodeRootInit(self.ark_mem, 0, cv_root)
                else:
                    flag = ARK.ARKodeRootInit(self.ark_mem, self.pData.dimRoot, cv_root)
                if flag < 0:
                    raise ARKODEError(flag, self.t)
            flag = SUNDIALS.SUNContext_PushErrHandler(self.ctx, cv_err, <void*>self.pData)
            if flag < 0:
                raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetUserData(self.ark_mem, <void*>self.pData)
            if flag < 0:
                raise ARKODEError(flag, self.t)
        else:
            # after an event: keep the method and its statistics, move to (t, y)
            flag = ARK.ARKodeReset(self.ark_mem, self.t, self.yTemp)
            if flag < 0:
                raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetUserData(self.ark_mem, <void*>self.pData)
            if flag < 0:
                raise ARKODEError(flag, self.t)

    cdef int _switch_table(self, double t, np.ndarray y, double tf, table) except -1:
        """Recreates the memory block with `table` at (t, y), keeping the last step size."""
        cdef double hlast = 0.0
        cdef int flag
        ARK.ARKodeGetLastStep(self.ark_mem, &hlast)
        self.store_statistics(ARK_TSTOP_RETURN)
        self.t = t
        self.y = y
        self._free_memory()
        self._active_table = table
        self.initialize_arkode()
        self.initialize_options()
        if hlast > 0.0:
            flag = ARK.ARKodeSetInitStep(self.ark_mem, hlast)   # the controller's last step, not a fresh estimate
            if flag < 0: raise ARKODEError(flag, t)
        flag = ARK.ARKodeSetStopTime(self.ark_mem, tf)
        if flag < 0: raise ARKODEError(flag, t)
        return 0

    cdef int _fallback_check(self, double t, np.ndarray y, double tf) except -1:
        """Failure-rate monitor (see the 'fallback_*' options). Returns 1 after switching the
        memory block to the fallback table at (t, y), else 0."""
        cdef long int nattempts = 0, nfails = 0
        table = self.options["fallback_table"]
        if table is None or self.fallback_time is not None or self.options["method"] != "implicit" \
                or table == self._active_table:
            return 0
        ARK.ARKodeGetNumStepAttempts(self.ark_mem, &nattempts)
        ARK.ARKodeGetNumStepSolveFails(self.ark_mem, &nfails)
        window = int(self.options["fallback_window"])
        if nattempts - self._fb_attempts0 < window:
            return 0
        # ARKODE counts a step failed by a recoverable rhs failure at a stage (an exception in
        # the problem's rhs: the FMU's own solver not converging at a trial point) together with
        # the steps Newton could not converge; only the latter say something about the method,
        # so the rhs failures of the window are taken out (at most one per failed step)
        nrhs = min(self.pData.nrhsfails - self._fb_rhsfails0, nfails - self._fb_fails0)
        rate = (nfails - self._fb_fails0 - nrhs) / float(nattempts - self._fb_attempts0)
        self._fb_attempts0 = nattempts
        self._fb_fails0 = nfails
        self._fb_rhsfails0 = self.pData.nrhsfails
        if rate <= float(self.options["fallback_conv_fail_rate"]):
            return 0
        self.log_message("ARKODE: %.0f %% of the last %d step attempts failed by nonlinear convergence (rhs failures "
                         "excluded) at t = %g; continuing with the table %s" % (100 * rate, window, t, table), NORMAL)
        self.fallback_time = t
        self._switch_table(t, y, tf, table)
        return 1

    cdef int _retry_after_failure(self, int flag, double t, double tf) except -1:
        """After a hard failure of the fallback table (repeated error-test or convergence
        failures), switch back to the user's table at the failure point, once. Returns 1 if
        the integration can go on, else 0."""
        cdef double tcur = t
        cdef N_Vector ycur
        cdef np.ndarray y
        if flag not in (ARK_ERR_FAILURE, ARK_CONV_FAILURE) or self.fallback_time is None \
                or self.fallback_return_time is not None:
            return 0
        ARK.ARKodeGetCurrentTime(self.ark_mem, &tcur)
        ycur = N_VNew_Serial(self.pData.dim, self.ctx)
        ARK.ARKodeGetDky(self.ark_mem, tcur, 0, ycur)
        y = nv2arr(ycur)
        N_VDestroy(ycur)
        self.log_message("ARKODE: the table %s failed at t = %g (flag %d); continuing with the table %s"
                         % (self._active_table, tcur, flag, self.options["table"] if self.options["table"]
                            else "of order %d" % self.options["order"]), NORMAL)
        self.fallback_return_time = tcur
        self._switch_table(tcur, y, tf, self.options["table"])
        return 1

    cpdef initialize_options(self):
        """Applies the options to the ARKODE memory (called on every (re)initialization)."""
        cdef int flag
        method = self.options["method"]

        # linear solver and Jacobian (implicit only)
        if method == "implicit":
            if self.options["linear_solver"] == "DENSE":
                if self.sun_matrix == NULL:
                    self.sun_matrix = SUNDIALS.SUNDenseMatrix(self.pData.dim, self.pData.dim, self.ctx)
                    self.sun_linearsolver = SUNDIALS.SUNLinSol_Dense(self.yTemp, self.sun_matrix, self.ctx)
                    flag = ARK.ARKodeSetLinearSolver(self.ark_mem, self.sun_linearsolver, self.sun_matrix)
                    if flag < 0:
                        raise ARKODEError(flag, self.t)
                if self.pData.JAC != NULL and self.options["usejac"]:
                    flag = ARK.ARKodeSetJacFn(self.ark_mem, cv_jac)
                else:
                    flag = ARK.ARKodeSetJacFn(self.ark_mem, NULL)
                if flag < 0:
                    raise ARKODEError(flag, self.t)
            elif self.options["linear_solver"] == "SPGMR":
                if self.sun_linearsolver == NULL:
                    self.sun_linearsolver = SUNDIALS.SUNLinSol_SPGMR(self.yTemp, PREC_NONE, self.options["maxkrylov"], self.ctx)
                    flag = ARK.ARKodeSetLinearSolver(self.ark_mem, self.sun_linearsolver, NULL)
                    if flag < 0:
                        raise ARKODEError(flag, self.t)
                if self.pData.JACV != NULL and self.options["usejac"]:
                    flag = ARK.ARKodeSetJacTimes(self.ark_mem, SUNDIALS.cv_spils_jtsetup_dummy, cv_jacv)
                else:
                    flag = ARK.ARKodeSetJacTimes(self.ark_mem, NULL, NULL)
                if flag < 0:
                    raise ARKODEError(flag, self.t)
            else:
                raise AssimuloException("ARKODE: 'linear_solver' must be DENSE or SPGMR.")

            flag = ARK.ARKodeSetMaxNonlinIters(self.ark_mem, int(self.options["max_nonlin_iters"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetNonlinConvCoef(self.ark_mem, float(self.options["nonlin_conv_coef"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetPredictorMethod(self.ark_mem, int(self.options["predictor"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetDeduceImplicitRhs(self.ark_mem, 1 if self.options["deduce_implicit_rhs"] else 0)
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetMaxConvFails(self.ark_mem, int(self.options["maxncf"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetLSetupFrequency(self.ark_mem, int(self.options["lsetup_frequency"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetJacEvalFrequency(self.ark_mem, int(self.options["jac_eval_frequency"]))
            if flag < 0: raise ARKODEError(flag, self.t)
            flag = ARK.ARKodeSetDeltaGammaMax(self.ark_mem, float(self.options["delta_gamma_max"]))
            if flag < 0: raise ARKODEError(flag, self.t)

        # method: a named table, or ARKODE's default table of the requested order. Set once per
        # memory block: ARKStep frees its Butcher tables on these calls and re-creates them on the
        # first initialization only, not on ARKodeReset (the option setters recreate the memory).
        if self._fresh_memory:
            table = self._active_table
            if table is not None:
                tname = str(table).encode("ascii")
                if method == "implicit":
                    flag = ARK.ARKStepSetTableName(self.ark_mem, tname, b"ARKODE_ERK_NONE")
                else:
                    flag = ARK.ARKStepSetTableName(self.ark_mem, b"ARKODE_DIRK_NONE", tname)
                if flag < 0:
                    raise ARKODEError(flag, self.t, "unknown Butcher table '%s'" % table)
            else:
                flag = ARK.ARKodeSetOrder(self.ark_mem, int(self.options["order"]))
                if flag < 0:
                    raise ARKODEError(flag, self.t, "'order' %s is not available" % self.options["order"])
            if int(self.options["interpolant_degree"]) >= 0:
                flag = ARK.ARKodeSetInterpolantDegree(self.ark_mem, int(self.options["interpolant_degree"]))
                if flag < 0: raise ARKODEError(flag, self.t)
            self._fresh_memory = 0

        # step-size adaptivity bounds
        flag = ARK.ARKodeSetSafetyFactor(self.ark_mem, float(self.options["safety"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMaxGrowth(self.ark_mem, float(self.options["max_growth"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMaxFirstGrowth(self.ark_mem, float(self.options["max_first_growth"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMaxEFailGrowth(self.ark_mem, float(self.options["max_efail_growth"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        if method == "implicit":
            flag = ARK.ARKodeSetMaxCFailGrowth(self.ark_mem, float(self.options["max_cfail_growth"]))
            if flag < 0: raise ARKODEError(flag, self.t)
        # step limits
        flag = ARK.ARKodeSetMaxErrTestFails(self.ark_mem, int(self.options["maxnef"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMaxNumSteps(self.ark_mem, int(self.options["maxsteps"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMaxStep(self.ark_mem, float(self.options["maxh"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        flag = ARK.ARKodeSetMinStep(self.ark_mem, float(self.options["minh"]))
        if flag < 0: raise ARKODEError(flag, self.t)
        # first step: the user's inith, else ARKODE's estimate; after an event the last step is
        # kept only with restart_h = "keep" (a full-size step over a fresh transient hides it
        # from the error estimator; measured on the native TRBDF2)
        if self.options["restart_h"] == "keep" and self.statistics["nsteps"] > 0:
            pass
        else:
            flag = ARK.ARKodeSetInitStep(self.ark_mem, float(self.options["inith"]))
            if flag < 0: raise ARKODEError(flag, self.t)

        # tolerances (atol may have changed with the nominals in handle_event)
        if self.nv_atol != NULL:
            N_VDestroy(self.nv_atol)
        self.nv_atol = self._arr2nv(self.options["atol"])
        flag = ARK.ARKodeSVtolerances(self.ark_mem, float(self.options["rtol"]), self.nv_atol)
        if flag < 0: raise ARKODEError(flag, self.t)

    def initialize_event_detection(self):
        if self.options["external_event_detection"]:
            def event_func(t, y):
                return 0, self.problem.state_events(t, y, self.sw)
            self.event_func = event_func
            _, self.g_old = self.event_func(self.t, self.y)
        else:
            def event_func(t, y):
                return self.problem.state_events(t, y, self.sw)
            self.event_func = event_func
            self.g_old = self.event_func(self.t, self.y)

    # ------------------------------------------------------------------ integration
    cpdef integrate(self, double t, np.ndarray[ndim=1, dtype=realtype] y, double tf, dict opts):
        cdef int flag, output_index
        cdef N_Vector yout
        cdef double tret = self.t, tout
        cdef list tr = [], yr = []
        cdef np.ndarray output_list

        yout = self._arr2nv(y)

        if opts["initialize"]:
            self.initialize_arkode()
            self.initialize_options()
            if self.options["external_event_detection"]:
                self.initialize_event_detection()

        flag = ARK.ARKodeSetStopTime(self.ark_mem, tf)
        if flag < 0:
            N_VDestroy(yout)
            raise ARKODEError(flag, t)

        if opts["report_continuously"] or opts["output_list"] is None:
            nsteps_call = 0
            while True:
                flag = ARK.ARKodeEvolve(self.ark_mem, tf, yout, &tret, ARK_ONE_STEP)
                if flag < 0:
                    if self._retry_after_failure(flag, tret, tf):
                        continue
                    self.store_statistics(ARK_TSTOP_RETURN)
                    N_VDestroy(yout)
                    raise ARKODEError(flag, tret)
                # ARKODE's maxsteps counts per ARKodeEvolve call; in one-step mode enforce it per
                # integrate call, as CVode's mxstep does for a call in normal mode
                nsteps_call += 1
                if nsteps_call > int(self.options["maxsteps"]):
                    self.store_statistics(ARK_TSTOP_RETURN)
                    N_VDestroy(yout)
                    raise ARKODEError(ARK_TOO_MUCH_WORK, tret)

                t = tret
                y = nv2arr(yout)
                if self.options["external_event_detection"] and self.problem_info["state_events"]:
                    told = self.t if (opts["report_continuously"] or not tr) else tr[-1]
                    event_flag, t, y = self.event_locator(told, t, y)
                    if event_flag == ID_PY_EVENT:
                        flag = ARK_ROOT_RETURN

                if opts["report_continuously"]:
                    try:
                        flag_initialize = self.report_solution(t, y, opts)
                    except Exception:
                        self.store_statistics(ARK_TSTOP_RETURN)
                        raise
                    if flag_initialize:
                        flag = CV_STEP_RETURN    # a step event: the integration is reinitialized
                else:
                    tr.append(t)
                    yr.append(y)

                if flag == ARK_ROOT_RETURN or flag == CV_STEP_RETURN:
                    self.store_statistics(flag)
                    flag = ID_EVENT
                    break
                if flag == ARK_TSTOP_RETURN:
                    flag = ID_COMPLETE
                    self.store_statistics(ARK_TSTOP_RETURN)
                    break
                self._fallback_check(t, y, tf)
        else:
            output_index = opts["output_index"]
            output_list = opts["output_list"][output_index:]
            for tout in output_list:
                flag = ARK.ARKodeEvolve(self.ark_mem, tout, yout, &tret, ARK_NORMAL)
                if flag < 0 and self._retry_after_failure(flag, tret, tf):
                    flag = ARK.ARKodeEvolve(self.ark_mem, tout, yout, &tret, ARK_NORMAL)
                if flag < 0:
                    self.store_statistics(ARK_TSTOP_RETURN)
                    N_VDestroy(yout)
                    raise ARKODEError(flag, tret)
                tr.append(tret)
                yr.append(nv2arr(yout))
                if flag == ARK_ROOT_RETURN:
                    self.store_statistics(ARK_ROOT_RETURN)
                    flag = ID_EVENT
                    if tret == tout:
                        output_index += 1
                    break
                if flag == ARK_TSTOP_RETURN:
                    self.store_statistics(ARK_TSTOP_RETURN)
                    flag = ID_COMPLETE
                    if tret == tout:
                        output_index += 1
                    break
                output_index += 1
                self._fallback_check(tret, yr[-1], tf)
            else:
                flag = ID_COMPLETE
                self.store_statistics(ARK_TSTOP_RETURN)
            opts["output_index"] = output_index

        N_VDestroy(yout)
        return flag, tr, yr

    cpdef step(self, double t, np.ndarray y, double tf, dict opts):
        cdef int flag
        cdef N_Vector yout
        cdef double tret = t
        yout = self._arr2nv(y)
        if opts["initialize"]:
            self.initialize_arkode()
            self.initialize_options()
        flag = ARK.ARKodeSetStopTime(self.ark_mem, tf)
        if flag < 0:
            raise ARKODEError(flag, t)
        flag = ARK.ARKodeEvolve(self.ark_mem, tf, yout, &tret, ARK_ONE_STEP)
        if flag < 0:
            raise ARKODEError(flag, tret)
        tr = tret
        yr = nv2arr(yout)
        if flag == ARK_ROOT_RETURN:
            flag = ID_EVENT
            self.store_statistics(ARK_ROOT_RETURN)
        if flag == ARK_TSTOP_RETURN:
            flag = ID_COMPLETE
            self.store_statistics(ARK_TSTOP_RETURN)
        N_VDestroy(yout)
        return flag, tr, yr

    cpdef state_event_info(self):
        if self.options["external_event_detection"]:
            return self._event_info
        cdef int* c_info
        cdef int flag
        c_info = <int*>malloc(self.pData.dimRoot * sizeof(int))
        for k in range(self.pData.dimRoot):
            c_info[k] = 0
        flag = ARK.ARKodeGetRootInfo(self.ark_mem, c_info)
        if flag < 0:
            free(c_info)
            raise ARKODEError(flag)
        event_info = [c_info[k] for k in range(self.pData.dimRoot)]
        free(c_info)
        return event_info

    def set_event_info(self, event_info):
        self._event_info = event_info

    cpdef np.ndarray interpolate(self, double t, int k = 0):
        """The dense output (ARKodeGetDky) at t within the last step; k = 0 for y, 1 for its derivative."""
        cdef int flag
        cdef N_Vector dky = N_VNew_Serial(self.pData.dim, self.ctx)
        flag = ARK.ARKodeGetDky(self.ark_mem, t, k, dky)
        if flag < 0:
            N_VDestroy(dky)
            raise ARKODEError(flag, t)
        res = nv2arr(dky)
        N_VDestroy(dky)
        return res

    cpdef get_local_errors(self):
        cdef int flag
        cdef N_Vector ele = N_VNew_Serial(self.pData.dim, self.ctx)
        flag = ARK.ARKodeGetEstLocalErrors(self.ark_mem, ele)
        if flag < 0:
            N_VDestroy(ele)
            raise ARKODEError(flag, self.t)
        res = nv2arr(ele)
        N_VDestroy(ele)
        return res

    cpdef get_error_weights(self):
        cdef int flag
        cdef N_Vector ew = N_VNew_Serial(self.pData.dim, self.ctx)
        flag = ARK.ARKodeGetErrWeights(self.ark_mem, ew)
        if flag < 0:
            N_VDestroy(ew)
            raise ARKODEError(flag, self.t)
        res = nv2arr(ew)
        N_VDestroy(ew)
        return res

    def get_weighted_local_errors(self):
        return np.abs(self.get_local_errors() * self.get_error_weights())

    cdef void store_statistics(self, int return_flag):
        """Adds ARKODE's counters (cumulative since the memory was created) as increments."""
        cdef long int nsteps = 0, nattempts = 0, netfails = 0, nfe = 0, nfi = 0, njevals = 0, nfevalsLS = 0
        cdef long int nniters = 0, nncfails = 0, nlinsetups = 0, ngevals = 0, njvevals = 0, nsolvefails = 0
        if self.ark_mem == NULL:
            raise ARKODEError(ARK_MEM_NULL)
        ARK.ARKodeGetNumSteps(self.ark_mem, &nsteps)
        ARK.ARKodeGetNumStepAttempts(self.ark_mem, &nattempts)
        ARK.ARKodeGetNumErrTestFails(self.ark_mem, &netfails)
        ARK.ARKodeGetNumRhsEvals(self.ark_mem, 0, &nfe)
        ARK.ARKodeGetNumRhsEvals(self.ark_mem, 1, &nfi)
        if self.options["method"] == "implicit":
            ARK.ARKodeGetNumNonlinSolvIters(self.ark_mem, &nniters)
            ARK.ARKodeGetNumNonlinSolvConvFails(self.ark_mem, &nncfails)
            ARK.ARKodeGetNumStepSolveFails(self.ark_mem, &nsolvefails)
            ARK.ARKodeGetNumLinSolvSetups(self.ark_mem, &nlinsetups)
            if self.options["linear_solver"] == "SPGMR":
                ARK.ARKodeGetNumJtimesEvals(self.ark_mem, &njvevals)
                ARK.ARKodeGetNumLinRhsEvals(self.ark_mem, &nfevalsLS)
            else:
                ARK.ARKodeGetNumJacEvals(self.ark_mem, &njevals)
                ARK.ARKodeGetNumLinRhsEvals(self.ark_mem, &nfevalsLS)
        if self.pData.ROOT != NULL:
            ARK.ARKodeGetNumGEvals(self.ark_mem, &ngevals)

        # the counters are cumulative over the life of the memory block (ARKodeReset keeps them):
        # store the difference to the last stored values
        cur = {"nsteps": nsteps, "nstepattempts": nattempts, "nerrfails": netfails, "nfcns": nfe + nfi,
               "nniters": nniters, "nnfails": nncfails, "nconvfails": nsolvefails, "nlus": nlinsetups,
               "njacs": njevals, "njacvecs": njvevals, "nfcnjacs": nfevalsLS, "nstatefcns": ngevals}
        last = self._last_counters
        for key, val in cur.items():
            prev = last.get(key, 0)
            if val < prev:
                prev = 0                 # ARKodeReset zeroes some counters (the root evaluations)
            self.statistics[key] += val - prev
        self._last_counters = cur
        if return_flag == ARK_ROOT_RETURN and not self.options["external_event_detection"]:
            self.statistics["nstateevents"] += 1

    def print_statistics(self, verbose=NORMAL):
        Explicit_ODE.print_statistics(self, verbose)
        log = lambda msg: self.log_message(msg, verbose)
        log("\nSolver options:\n")
        log(" Solver                  : ARKODE (%s, %s)" % (self.options["method"],
            self.options["table"] if self.options["table"] else "order %d" % self.options["order"]))
        if self.fallback_time is not None:
            log(" Fallback                : %s from t = %g%s" % (self.options["fallback_table"], self.fallback_time,
                "" if self.fallback_return_time is None else ", back from t = %g" % self.fallback_return_time))
        if self.options["method"] == "implicit":
            log(" Linear solver           : " + self.options["linear_solver"])
        log(" Tolerances (absolute)   : " + str(self._compact_tol(self.options["atol"])))
        log(" Tolerances (relative)   : " + str(self.options["rtol"]))
        log("")

    # ------------------------------------------------------------------ options as properties
    def _set_atol(self, atol):
        atol = set_type_shape_array(atol)
        if len(atol) == 1:
            atol = atol * np.ones(self.pData.dim)
        elif len(atol) != self.pData.dim:
            raise AssimuloException("atol must be of length one or same as the dimension of the problem.")
        if (atol < 0.0).any():
            raise AssimuloException("The absolute tolerance must be positive.")
        self.options["atol"] = atol
    def _get_atol(self):
        return self.options["atol"]
    atol = property(_get_atol, _set_atol)

    def _set_rtol(self, rtol):
        rtol = float(rtol)
        if rtol < 0.0:
            raise AssimuloException("The relative tolerance must be positive.")
        self.options["rtol"] = rtol
    def _get_rtol(self):
        return self.options["rtol"]
    rtol = property(_get_rtol, _set_rtol)

    def _set_maxh(self, maxh):
        self.options["maxh"] = float(maxh) if maxh else 0.0
    def _get_maxh(self):
        return self.options["maxh"]
    maxh = property(_get_maxh, _set_maxh)

    def _set_minh(self, minh):
        self.options["minh"] = float(minh)
    def _get_minh(self):
        return self.options["minh"]
    minh = property(_get_minh, _set_minh)

    def _set_inith(self, inith):
        self.options["inith"] = float(inith)
    def _get_inith(self):
        return self.options["inith"]
    inith = property(_get_inith, _set_inith)

    def _set_maxsteps(self, n):
        self.options["maxsteps"] = int(n)
    def _get_maxsteps(self):
        return self.options["maxsteps"]
    maxsteps = property(_get_maxsteps, _set_maxsteps)

    def _set_fallback_table(self, table):
        self.options["fallback_table"] = str(table) if table else None
    def _get_fallback_table(self):
        return self.options["fallback_table"]
    fallback_table = property(_get_fallback_table, _set_fallback_table)

    def _set_fallback_conv_fail_rate(self, rate):
        rate = float(rate)
        if not 0.0 < rate <= 1.0:
            raise AssimuloException("fallback_conv_fail_rate must be in (0, 1].")
        self.options["fallback_conv_fail_rate"] = rate
    def _get_fallback_conv_fail_rate(self):
        return self.options["fallback_conv_fail_rate"]
    fallback_conv_fail_rate = property(_get_fallback_conv_fail_rate, _set_fallback_conv_fail_rate)

    def _set_fallback_window(self, n):
        n = int(n)
        if n < 1:
            raise AssimuloException("fallback_window must be at least 1.")
        self.options["fallback_window"] = n
    def _get_fallback_window(self):
        return self.options["fallback_window"]
    fallback_window = property(_get_fallback_window, _set_fallback_window)

    def _set_usejac(self, jac):
        self.options["usejac"] = bool(jac)
    def _get_usejac(self):
        return self.options["usejac"]
    usejac = property(_get_usejac, _set_usejac)

    def _set_linear_solver(self, ls):
        ls = str(ls).upper()
        if ls not in ("DENSE", "SPGMR"):
            raise AssimuloException("'linear_solver' must be DENSE or SPGMR.")
        if ls != self.options["linear_solver"]:
            self._free_memory()
        self.options["linear_solver"] = ls
    def _get_linear_solver(self):
        return self.options["linear_solver"]
    linear_solver = property(_get_linear_solver, _set_linear_solver)

    def _set_maxkrylov(self, n):
        self.options["maxkrylov"] = int(n)
    def _get_maxkrylov(self):
        return self.options["maxkrylov"]
    maxkrylov = property(_get_maxkrylov, _set_maxkrylov)

    def _set_external_event_detection(self, v):
        v = bool(v)
        if v != self.options["external_event_detection"]:
            self._free_memory()                 # the root function set-up differs
        self.options["external_event_detection"] = v
    def _get_external_event_detection(self):
        return self.options["external_event_detection"]
    external_event_detection = property(_get_external_event_detection, _set_external_event_detection)

    def _set_method(self, m):
        m = str(m).lower()
        if m not in ("implicit", "explicit"):
            raise AssimuloException("'method' must be 'implicit' or 'explicit'.")
        if m != self.options["method"]:
            self._free_memory()
        self.options["method"] = m
    def _get_method(self):
        return self.options["method"]
    method = property(_get_method, _set_method)

    def _set_order(self, q):
        if int(q) != self.options["order"]:
            self._free_memory()
        self.options["order"] = int(q)
    def _get_order(self):
        return self.options["order"]
    order = property(_get_order, _set_order)

    def _set_table(self, name):
        name = None if name is None else str(name)
        if name != self.options["table"]:
            self._free_memory()
        self.options["table"] = name
    def _get_table(self):
        return self.options["table"]
    table = property(_get_table, _set_table)

    def _set_predictor(self, p):
        self.options["predictor"] = int(p)
    def _get_predictor(self):
        return self.options["predictor"]
    predictor = property(_get_predictor, _set_predictor)

    def _set_nonlin_conv_coef(self, c):
        self.options["nonlin_conv_coef"] = float(c)
    def _get_nonlin_conv_coef(self):
        return self.options["nonlin_conv_coef"]
    nonlin_conv_coef = property(_get_nonlin_conv_coef, _set_nonlin_conv_coef)

    def _set_max_nonlin_iters(self, n):
        self.options["max_nonlin_iters"] = int(n)
    def _get_max_nonlin_iters(self):
        return self.options["max_nonlin_iters"]
    max_nonlin_iters = property(_get_max_nonlin_iters, _set_max_nonlin_iters)

    def _set_deduce_implicit_rhs(self, v):
        self.options["deduce_implicit_rhs"] = bool(v)
    def _get_deduce_implicit_rhs(self):
        return self.options["deduce_implicit_rhs"]
    deduce_implicit_rhs = property(_get_deduce_implicit_rhs, _set_deduce_implicit_rhs)

    def _set_interpolant_degree(self, d):
        if int(d) != self.options["interpolant_degree"]:
            self._free_memory()
        self.options["interpolant_degree"] = int(d)
    def _get_interpolant_degree(self):
        return self.options["interpolant_degree"]
    interpolant_degree = property(_get_interpolant_degree, _set_interpolant_degree)

    def _set_maxncf(self, n):
        self.options["maxncf"] = int(n)
    def _get_maxncf(self):
        return self.options["maxncf"]
    maxncf = property(_get_maxncf, _set_maxncf)

    def _set_maxnef(self, n):
        self.options["maxnef"] = int(n)
    def _get_maxnef(self):
        return self.options["maxnef"]
    maxnef = property(_get_maxnef, _set_maxnef)

    def _set_lsetup_frequency(self, n):
        self.options["lsetup_frequency"] = int(n)
    def _get_lsetup_frequency(self):
        return self.options["lsetup_frequency"]
    lsetup_frequency = property(_get_lsetup_frequency, _set_lsetup_frequency)

    def _set_jac_eval_frequency(self, n):
        self.options["jac_eval_frequency"] = int(n)
    def _get_jac_eval_frequency(self):
        return self.options["jac_eval_frequency"]
    jac_eval_frequency = property(_get_jac_eval_frequency, _set_jac_eval_frequency)

    def _set_delta_gamma_max(self, v):
        self.options["delta_gamma_max"] = float(v)
    def _get_delta_gamma_max(self):
        return self.options["delta_gamma_max"]
    delta_gamma_max = property(_get_delta_gamma_max, _set_delta_gamma_max)

    def _set_safety(self, v):
        self.options["safety"] = float(v)
    def _get_safety(self):
        return self.options["safety"]
    safety = property(_get_safety, _set_safety)

    def _set_max_growth(self, v):
        self.options["max_growth"] = float(v)
    def _get_max_growth(self):
        return self.options["max_growth"]
    max_growth = property(_get_max_growth, _set_max_growth)

    def _set_max_first_growth(self, v):
        self.options["max_first_growth"] = float(v)
    def _get_max_first_growth(self):
        return self.options["max_first_growth"]
    max_first_growth = property(_get_max_first_growth, _set_max_first_growth)

    def _set_max_efail_growth(self, v):
        self.options["max_efail_growth"] = float(v)
    def _get_max_efail_growth(self):
        return self.options["max_efail_growth"]
    max_efail_growth = property(_get_max_efail_growth, _set_max_efail_growth)

    def _set_max_cfail_growth(self, v):
        self.options["max_cfail_growth"] = float(v)
    def _get_max_cfail_growth(self):
        return self.options["max_cfail_growth"]
    max_cfail_growth = property(_get_max_cfail_growth, _set_max_cfail_growth)

    def _set_restart_h(self, v):
        v = str(v).lower()
        if v not in ("estimate", "keep"):
            raise AssimuloException("'restart_h' must be 'estimate' or 'keep'.")
        self.options["restart_h"] = v
    def _get_restart_h(self):
        return self.options["restart_h"]
    restart_h = property(_get_restart_h, _set_restart_h)


class ARKODEError(Exception):
    """An ARKODE return flag with its textual meaning."""
    msg = {ARK_TOO_MUCH_WORK: "The solver took the maximum number of internal steps but could not reach tout.",
           ARK_TOO_MUCH_ACC: "The solver could not satisfy the accuracy demanded for some internal step.",
           ARK_ERR_FAILURE: "Error test failures occurred too many times during one internal step or the minimum step size was reached.",
           ARK_CONV_FAILURE: "Convergence test failures occurred too many times during one internal step or the minimum step size was reached.",
           ARK_LINIT_FAIL: "The linear solver's initialization function failed.",
           ARK_LSETUP_FAIL: "The linear solver's setup function failed in an unrecoverable manner.",
           ARK_LSOLVE_FAIL: "The linear solver's solve function failed in an unrecoverable manner.",
           ARK_RHSFUNC_FAIL: "The right-hand side function failed in an unrecoverable manner.",
           ARK_FIRST_RHSFUNC_ERR: "The right-hand side function failed at the first call.",
           ARK_REPTD_RHSFUNC_ERR: "The right-hand side function had repeated recoverable errors.",
           ARK_UNREC_RHSFUNC_ERR: "The right-hand side function had a recoverable error, but no recovery is possible.",
           ARK_RTFUNC_FAIL: "The rootfinding function failed in an unrecoverable manner.",
           ARK_MEM_FAIL: "A memory allocation failed.",
           ARK_MEM_NULL: "The ARKODE memory was NULL.",
           ARK_ILL_INPUT: "One of the function inputs is illegal.",
           ARK_TOO_CLOSE: "The output and initial times are too close to each other.",
           ARK_INTERP_FAIL: "The interpolation module failed.",
           ARK_STEPPER_UNSUPPORTED: "The stepper does not support this operation."}

    def __init__(self, value, t=0.0, extra=None):
        self.value = value
        self.t = t
        self.extra = extra

    def __str__(self):
        text = self.msg.get(self.value, "Unknown error (%d)." % self.value)
        if self.extra:
            text += " " + str(self.extra)
        return "ARKODE failed with flag %d at time %g: %s" % (self.value, self.t, text)

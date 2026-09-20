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

import numpy as np
import pytest

from assimulo.problem import Explicit_Problem
from assimulo.solvers import Radau5ODE
from assimulo.exception import AssimuloException

arkode = pytest.importorskip("assimulo.solvers.arkode")
ARKODE, ARKODEError = arkode.ARKODE, arkode.ARKODEError


def vanderpol(mu=1e6):
    def f(t, y):
        return np.array([y[1], mu * ((1. - y[0]**2) * y[1] - y[0])])

    def jac(t, y):
        return np.array([[0., 1.], [mu * (-2. * y[0] * y[1] - 1.), mu * (1. - y[0]**2)]])
    mod = Explicit_Problem(f, [2.0, -0.6])
    mod.jac = jac
    return mod


class Test_ARKODE:

    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    def test_prothero_robinson(self, order):
        """Stiff linear problem with a known solution: the error follows the tolerance at every order."""
        lam = -1000.0
        mod = Explicit_Problem(lambda t, y: np.array([lam * (y[0] - np.cos(t)) - np.sin(t)]), [1.0])
        mod.jac = lambda t, y: np.array([[lam]])
        errs = {}
        for tol in (1e-4, 1e-6, 1e-8):
            sim = ARKODE(mod)
            sim.order = order
            sim.maxsteps = 100000        # order 2 at rtol 1e-8
            sim.rtol = tol
            sim.atol = 1e-2 * tol
            sim.verbosity = 50
            t, y = sim.simulate(10.0)
            errs[tol] = abs(y[-1][0] - np.cos(10.0))
            assert sim.statistics["nfcnjacs"] == 0            # the user Jacobian was used
            assert sim.statistics["nstepattempts"] >= sim.statistics["nsteps"]
        # DIRK methods show order reduction on this problem (stage order 1-2), so the global
        # error is not below rtol at every order; it follows the tolerance within a factor
        assert errs[1e-4] < 1e-3 and errs[1e-6] < 1e-4 and errs[1e-8] < 1e-6
        assert errs[1e-8] < errs[1e-4]

    def test_vanderpol_against_radau5(self):
        mod = vanderpol()
        ref = Radau5ODE(mod)
        ref.rtol = 1e-10
        ref.atol = 1e-12
        ref.verbosity = 50
        _, yref = ref.simulate(2.0)
        sim = ARKODE(mod)
        sim.rtol = 1e-6
        sim.atol = 1e-8
        sim.verbosity = 50
        _, y = sim.simulate(2.0)
        assert y[-1][0] == pytest.approx(yref[-1][0], abs=1e-4)
        assert sim.statistics["nfcnjacs"] == 0
        assert sim.statistics["nsteps"] < 2000
        # with deduce_implicit_rhs the stage derivatives cost no rhs calls beyond the Newton iterations
        assert sim.statistics["nfcns"] <= sim.statistics["nniters"] + 2 * sim.statistics["nstepattempts"]

    def test_table_by_name(self):
        mod = vanderpol()
        sim = ARKODE(mod)
        sim.table = "ARKODE_TRBDF2_3_3_2"
        sim.rtol = 1e-6
        sim.atol = 1e-8
        sim.verbosity = 50
        _, y = sim.simulate(2.0)
        assert sim.statistics["nsteps"] > 0
        sim = ARKODE(mod)
        sim.table = "ARKODE_NO_SUCH_TABLE"
        sim.verbosity = 50
        with pytest.raises(ARKODEError, match="Butcher table"):
            sim.simulate(2.0)

    def test_no_jac(self):
        mod = vanderpol(mu=100.0)
        del mod.jac
        sim = ARKODE(mod)
        sim.verbosity = 50
        assert not sim.usejac
        sim.simulate(1.0)
        assert sim.statistics["nfcnjacs"] == 2 * sim.statistics["njacs"]   # ARKODE's difference quotients, one per column

    def test_explicit(self):
        """Explicit RK on a non-stiff problem: no Jacobian, no Newton, high order."""
        mod = Explicit_Problem(lambda t, y: np.array([y[1], -y[0]]), [1.0, 0.0])
        sim = ARKODE(mod)
        sim.method = "explicit"
        sim.order = 5
        sim.rtol = 1e-8
        sim.atol = 1e-10
        sim.verbosity = 50
        t, y = sim.simulate(10.0)
        assert y[-1][0] == pytest.approx(np.cos(10.0), abs=1e-6)
        assert sim.statistics["njacs"] == 0 and sim.statistics["nniters"] == 0 and sim.statistics["nlus"] == 0
        assert sim.statistics["nsteps"] < 400

    def test_interpolate_and_ncp(self):
        """Dense output: communication points are interpolated, not stepped to."""
        mod = Explicit_Problem(lambda t, y: np.array([-y[0], -100. * (y[1] - np.cos(t))]), [1.0, 1.0])
        sim = ARKODE(mod)
        sim.rtol = 1e-8
        sim.atol = 1e-10
        sim.verbosity = 50
        t, y = sim.simulate(3.0, 300)
        assert len(t) == 301
        assert np.abs(np.array(y)[:, 0] - np.exp(-np.array(t))).max() < 1e-6
        sim.reset()
        sim.report_continuously = True
        t2, y2 = sim.simulate(3.0, 300)
        assert len(t2) == 301
        assert np.abs(np.array(y2)[:, 0] - np.exp(-np.array(t2))).max() < 1e-6

    @pytest.mark.parametrize("external", [False, True])
    def test_event_localizer(self, extended_problem, external):
        sim = ARKODE(extended_problem)
        sim.external_event_detection = external
        sim.verbosity = 50
        sim.report_continuously = True
        t, y = sim.simulate(10.0, 1000)
        assert y[-1][0] == pytest.approx(8.0)
        assert y[-1][1] == pytest.approx(3.0)
        assert y[-1][2] == pytest.approx(2.0)

    def test_time_event(self):
        events = [1.0, 2.0, 2.5, 3.0]
        seen = []

        def time_events(t, y, sw):
            for ev in events:
                if t < ev:
                    return ev
            return None

        def handle_event(solver, event_info):
            solver.y += 1.0
            seen.append(solver.t)
            assert event_info[0] == []
            assert event_info[1]
        mod = Explicit_Problem(lambda t, y: np.array([1.0]), [0.0])
        mod.time_events = time_events
        mod.handle_event = handle_event
        sim = ARKODE(mod)
        sim.verbosity = 50
        t, y = sim.simulate(5.0, 100)
        assert [pytest.approx(e) for e in events] == seen
        assert y[-1][0] == pytest.approx(5.0 + 4.0)

    def test_switches(self):
        state_events = lambda t, x, sw: np.array([x[0] - 1.])

        def handle_event(solver, event_info):
            solver.sw = [False]
        mod = Explicit_Problem(lambda t, x, sw: np.array([1.0]), [0.0])
        mod.sw0 = [True]
        mod.state_events = state_events
        mod.handle_event = handle_event
        sim = ARKODE(mod)
        sim.verbosity = 50
        assert sim.sw[0]
        sim.simulate(3)
        assert not sim.sw[0]
        assert sim.statistics["nstateevents"] == 1

    def test_statistics_accumulate_over_events(self):
        """ARKODE's counters are cumulative over the memory block; the statistics must not double count."""
        mod = Explicit_Problem(lambda t, y: np.array([1.0]), [0.0])
        mod.state_events = lambda t, y, sw: np.array([y[0] - 1.0, y[0] - 2.0])
        mod.handle_event = lambda solver, info: None
        sim = ARKODE(mod)
        sim.verbosity = 50
        sim.simulate(3.0)
        assert sim.statistics["nstateevents"] == 2
        assert 0 < sim.statistics["nsteps"] < 200
        assert sim.statistics["nfcns"] < 50 * sim.statistics["nsteps"]

    def test_change_method_between_runs(self):
        """Method-defining options between two simulations recreate the ARKODE memory."""
        mod = Explicit_Problem(lambda t, y: np.array([-y[0], -100. * (y[1] - np.cos(t))]), [1.0, 1.0])
        sim = ARKODE(mod)
        sim.verbosity = 50
        sim.simulate(1.0, 10)
        sim.order = 3
        sim.simulate(2.0, 10)
        sim.table = "ARKODE_TRBDF2_3_3_2"
        sim.reset()
        sim.simulate(1.0, 10)
        sim.method = "explicit"
        sim.table = None
        sim.reset()
        t, y = sim.simulate(1.0, 10)
        assert y[-1][0] == pytest.approx(np.exp(-1.0), abs=1e-5)

    def test_maxsteps(self):
        sim = ARKODE(vanderpol())
        sim.verbosity = 50
        sim.maxsteps = 5
        with pytest.raises(ARKODEError, match="maximum number of internal steps"):
            sim.simulate(2.0)

    def test_options_validation(self):
        sim = ARKODE(vanderpol())
        with pytest.raises(AssimuloException):
            sim.rtol = -1.0
        with pytest.raises(AssimuloException):
            sim.atol = [-1.0, 1.0]
        with pytest.raises(AssimuloException):
            sim.method = "imex"
        with pytest.raises(AssimuloException):
            sim.linear_solver = "SPARSE"
        sim.maxh = None
        assert sim.maxh == 0.0
        sim.maxh = 0.1
        assert sim.maxh == 0.1
        sim.predictor = 2
        sim.nonlin_conv_coef = 0.05
        sim.deduce_implicit_rhs = False
        sim.restart_h = "keep"
        sim.lsetup_frequency = 5
        sim.jac_eval_frequency = 10
        sim.delta_gamma_max = 0.1
        assert sim.predictor == 2 and sim.nonlin_conv_coef == 0.05 and not sim.deduce_implicit_rhs
        sim.verbosity = 50
        sim.simulate(0.01)                # the properties reach ARKODE without an error

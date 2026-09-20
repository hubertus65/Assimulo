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
        sim.max_growth = 5.0
        sim.safety = 0.9
        assert sim.predictor == 2 and sim.nonlin_conv_coef == 0.05 and not sim.deduce_implicit_rhs
        sim.verbosity = 50
        sim.simulate(0.01)                # the properties reach ARKODE without an error


def sliding_mode(K=1e4):
    """A rhs with a jump the solution rides on (an all-or-nothing switch without an event
    indicator, as in the absorption-plant FMU): pushed up below y1 = 0, pulled down stiffly
    above it. Newton on the implicit stages fails on a large share of the steps whatever the
    Jacobian. y2 is smooth: y2 = (cos t + sin t - exp(-t)) / 2."""
    def f(t, y):
        return np.array([1.0 if y[0] < 0.0 else -K * y[0] - 1.0, -y[1] + np.cos(t)])

    def jac(t, y):
        return np.array([[0.0 if y[0] < 0.0 else -K, 0.0], [0.0, -1.0]])
    mod = Explicit_Problem(f, [-0.5, 0.0])
    mod.jac = jac
    return mod


class Test_ARKODE_fallback:
    def _solver(self, **opts):
        s = ARKODE(sliding_mode())
        s.rtol = s.atol = 1e-3
        s.verbosity = 50
        for k, v in opts.items():
            setattr(s, k, v)
        return s

    def test_defaults(self):
        s = self._solver()
        assert s.fallback_table == "ARKODE_TRBDF2_3_3_2"
        assert s.fallback_conv_fail_rate == 0.25
        assert s.fallback_window == 50
        assert s.fallback_time is None
        with pytest.raises(AssimuloException):
            s.fallback_conv_fail_rate = 1.5
        with pytest.raises(AssimuloException):
            s.fallback_window = 0
        s.fallback_table = None
        assert s.fallback_table is None

    def test_fallback_triggers_and_continues(self):
        s = self._solver(fallback_conv_fail_rate=0.2, fallback_window=10)
        t, y = s.simulate(2.0)
        assert s.fallback_time is not None and 0.0 < s.fallback_time < 2.0
        assert s.statistics["nconvfails"] > 0
        # the smooth state is right; the sliding one has reached the switch and never left it
        # upwards (above 0 it is pulled down stiffly)
        assert abs(y[-1, 1] - (np.cos(2.0) + np.sin(2.0) - np.exp(-2.0)) / 2) < 5e-3
        assert -0.5 < y[-1, 0] < 1e-3
        assert t[-1] == 2.0
        # a second simulation starts again from the user's table and falls back again
        s.reset()
        s.simulate(2.0)
        assert s.fallback_time is not None

    def test_fallback_disabled(self):
        s = self._solver(fallback_conv_fail_rate=0.2, fallback_window=10, fallback_table=None)
        t, y = s.simulate(2.0)
        assert s.fallback_time is None
        assert abs(y[-1, 1] - (np.cos(2.0) + np.sin(2.0) - np.exp(-2.0)) / 2) < 5e-3

    def test_no_fallback_on_a_smooth_problem(self):
        s = ARKODE(vanderpol())
        s.verbosity = 50
        s.simulate(2.0)
        assert s.fallback_time is None

    def test_fallback_with_output_points(self):
        # ARK_NORMAL mode: the rate is checked at the communication points over at least
        # 'fallback_window' attempts, so the failures near the switch are diluted -- lower threshold
        s = self._solver(fallback_conv_fail_rate=0.1, fallback_window=10)
        t, y = s.simulate(2.0, 400)
        assert len(t) == 401
        assert s.fallback_time is not None
        assert abs(y[-1, 1] - (np.cos(2.0) + np.sin(2.0) - np.exp(-2.0)) / 2) < 5e-3

    def test_fallback_not_for_the_fallback_table(self):
        s = self._solver(fallback_conv_fail_rate=0.2, fallback_window=10, table="ARKODE_TRBDF2_3_3_2")
        s.simulate(2.0)
        assert s.fallback_time is None


def kepler(e=0.3):
    """y = [q1, q2, p1, p2], H = |p|^2/2 - 1/|q|; a separable Hamiltonian system."""
    def f(t, y):
        q1, q2, p1, p2 = y
        r3 = (q1 * q1 + q2 * q2) ** 1.5
        return np.array([p1, p2, -q1 / r3, -q2 / r3])
    return Explicit_Problem(f, [1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])


def energy(y):
    return 0.5 * (y[:, 2]**2 + y[:, 3]**2) - 1 / np.sqrt(y[:, 0]**2 + y[:, 1]**2)


class Test_ARKODE_symplectic:
    def test_options(self):
        s = ARKODE(kepler())
        s.method = "symplectic"
        assert s.method == "symplectic"
        with pytest.raises(AssimuloException):
            s.simulate(1.0)                 # q_states missing
        s.q_states = [0, 1]
        with pytest.raises(AssimuloException):
            s.simulate(1.0)                 # fixed_h missing
        s.fixed_h = 0.01
        s.q_states = [0, 1, 2, 3]
        with pytest.raises(AssimuloException):
            s.simulate(1.0)                 # all states positions
        s.q_states = np.array([True, True, False, False])
        s.verbosity = 50
        t, y = s.simulate(1.0)
        assert t[-1] == 1.0

    def test_energy_is_conserved(self):
        """Over 100 orbits the symplectic method's energy error stays bounded at the level of
        one step's truncation error; the explicit RK at the same step count drifts."""
        tf = 100 * 2 * np.pi
        s = ARKODE(kepler()); s.method = "symplectic"; s.order = 4; s.q_states = [0, 1]; s.fixed_h = 0.02; s.verbosity = 50
        s.maxsteps = 10**6
        t, y = s.simulate(tf, 1000)
        dH = np.abs(energy(y) - energy(y[:1]))
        assert dH.max() < 1e-5
        assert dH[-100:].max() < 2 * dH[:100].max()      # no drift
        assert s.statistics["nfcns"] == 2 * 4 * s.statistics["nsteps"]   # 4 stages, f1 and f2 each a full rhs

    def test_tables_and_orders(self):
        for order, table in ((2, None), (6, None), (None, "ARKODE_SPRK_MCLACHLAN_4_4"), (None, "ARKODE_SPRK_YOSHIDA_6_8")):
            s = ARKODE(kepler()); s.method = "symplectic"; s.q_states = [0, 1]; s.fixed_h = 0.01; s.verbosity = 50
            if order: s.order = order
            if table: s.table = table
            t, y = s.simulate(2 * np.pi, 100)
            assert np.abs(energy(y) - energy(y[:1])).max() < (1e-3 if order == 2 else 1e-6), (order, table)
        s = ARKODE(kepler()); s.method = "symplectic"; s.q_states = [0, 1]; s.fixed_h = 0.01; s.table = "ARKODE_SPRK_NO_SUCH"
        with pytest.raises(ARKODEError):
            s.simulate(1.0)

    def test_state_event(self):
        """A pendulum-like oscillator whose position crossing zero is an event; the rootfinding
        works on SPRKStep's interpolant and the run restarts on the fixed step."""
        def f(t, y, sw):
            return np.array([y[1], -y[0]])
        def g(t, y, sw):
            return np.array([y[0]])
        def handle(solver, info):
            solver.sw[0] = not solver.sw[0]
        mod = Explicit_Problem(f, [1.0, 0.0], sw0=[True]); mod.state_events = g; mod.handle_event = handle
        s = ARKODE(mod); s.method = "symplectic"; s.q_states = [0]; s.fixed_h = 0.01; s.verbosity = 50
        t, y = s.simulate(3 * np.pi, 300)
        assert s.statistics["nstateevents"] == 3
        assert abs(y[-1, 0] - np.cos(3 * np.pi)) < 1e-3

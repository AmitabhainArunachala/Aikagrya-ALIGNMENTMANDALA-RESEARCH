from aikagrya.mmip.schedules import cosine_schedule, exp_alpha_schedule


def test_schedules_move():
    T = 1000
    vals_tau = [cosine_schedule(t, T, 0.1, 0.02) for t in range(0, T, 50)]
    vals_a = [exp_alpha_schedule(t, T, 0.6, 0.9995) for t in range(0, T, 50)]
    assert len(set(round(v, 6) for v in vals_tau[-10:])) > 3
    assert len(set(round(v, 6) for v in vals_a[-10:])) > 3



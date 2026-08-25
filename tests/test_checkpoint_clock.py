import embodied


def test_local_clock_zero_disables_periodic_calls():
  clock = embodied.LocalClock(0)
  assert not clock()
  assert not clock()


def test_local_clock_negative_means_every_call():
  clock = embodied.LocalClock(-1)
  assert clock()
  assert clock()

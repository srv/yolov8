#!/usr/bin/env python3

import time

class NonBlockingDelay():
  """ Non blocking delay class """
  def __init__(self):
    self._timestamp = 0
    self._delay = 0

  def timeout(self):
    """ Check if time is up """
    return ((millis() - self._timestamp) > self._delay)

  def delay_ms(self, delay):
    """ Non blocking delay in ms """
    self._timestamp = millis()
    self._delay = delay

def millis():
  """ Get millis """
  return int(time.time() * 1000)

def delay_ms(delay):
  """ Blocking delay in ms """
  t0 = millis()
  while (millis() - t0) < delay:
    pass

import numpy as np

def shuffle_in_unison(a, b):
  cur_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(cur_state)
  np.random.shuffle(b)
  return a, b

a = [[0,1],[2,3],[4,5],[6,7],[8,9]]
b = [[10,11],[12,13],[14,15],[16,17],[18,19]]
print ('init {} {}'.format(a,b))
a,b = shuffle_in_unison(a,b)
print ('now {} {}'.format(a,b))

import sys
sys.path.append('../')
from egnn import test_egnn_equivariance
from pool import test_pool_functionality

# * run all available tests
test_egnn_equivariance(1000, True)
test_pool_functionality(500, True)

print ('[test.py] all tests passed!')
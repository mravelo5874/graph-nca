import sys
sys.path.append('../')
from egnn import test_egnn_equivariance
from pool import test_pool_functionality
from models import test_model_training
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tests', type=str, default=10, help='number of tests to run per function')
args = parser.parse_args()

# * run all available tests
n_tests = int(args.tests)

print ('[test.py] running egnn equivariance tests')
test_egnn_equivariance(n_tests, True)

print ('[test.py] running pool functionality tests')
test_pool_functionality(n_tests, True)

print ('[test.py] running model train tests')
test_model_training(n_tests, True)

print ('[test.py] all tests passed!')
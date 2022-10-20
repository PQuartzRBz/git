import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', default="test.log", type=str)
parser.add_argument('--num_1', default=20, type=int)
parser.add_argument('--num_2', default=10, type=int)
args = parser.parse_args()


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    return x / y

logging.basicConfig(filename='test.log', level=logging.debug,
                    format='%(asctime)s:%(levelname)s:%(message)s')

num_1 = 20
num_2 = 10

add_result = add(num_1, num_2)
logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

sub_result = subtract(num_1, num_2)
logging.warning('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

mul_result = multiply(num_1, num_2)
logging.error('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

div_result = divide(num_1, num_2)
logging.critical('Div: {} / {} = {}'.format(num_1, num_2, div_result))

th = 10+6
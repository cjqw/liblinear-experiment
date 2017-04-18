from tools.partition import *

# The file name of trainning data
TRAIN_DATA_SET = 'train.txt'
# The file name of test data
TEST_DATA_SET = 'test.txt'
# The path to store .model files
MODEL_PATH = 'models/'
# The path to store input files
DATA_PATH = 'data/'

# Max quantity of input & test data items
# Set to small number for debug
# Set it to 0 or below to forbid dropping data
TRAIN_DATA = 0

# Maximum class when training Min-Max models with
# randomized partition function
MAX_CLASS = 3

# Whether trust the post models or not
MEMORIZE = False

# which partition function to use
PARTITION_FUNCTION = getRandClassFunc(MAX_CLASS)

# which algorithm to use
# 1: brute linear svm classifier
# 2: min-max algorithm
# 3: parallelized min-max algorithm

BRUTE_ALGORITHM = 1
MIN_MAX_ALGORITHM = 2
PARALLELIZED_MIN_MAX = 3

ALGORITHM = BRUTE_ALGORITHM

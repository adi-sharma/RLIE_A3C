#mode = "EMA"
mode = "Shooter"

if mode == "Shooter":
    #for Shooter DB
    int2tags = ['shooterName', 'killedNum', 'woundedNum', 'city']
    tags2int = {'TAG':0,\
    'shooterName':1, \
    'killedNum':2, \
    'woundedNum':3, \
    'city' : 4 }

elif mode == "EMA":
    # for EMA
    int2tags = \
    ['Affected-Food-Product',\
    'Produced-Location',\
    'Adulterant(s)']
    tags2int = \
    {'TAG':0,\
    'Affected-Food-Product':1, \
    'Produced-Location':2, \
    'Adulterant(s)':3}
    int2citationFeilds = ['Authors', 'Date', 'Title', 'Source']
    generic = [
        "city", "centre", "county", "street", "road", "and", "in", "town",
        "village"
    ]

NUM_ENTITIES = len(int2tags)
WORD_LIMIT = 1000
CONTEXT_LENGTH = 3
STATE_SIZE = 4 * NUM_ENTITIES + 1 + 2 * CONTEXT_LENGTH * NUM_ENTITIES
STOP_ACTION = NUM_ENTITIES
IGNORE_ALL = STOP_ACTION + 1
ACCEPT_ALL = 999  #arbitrary

# Keep the following same as the agent values
PORT = 7000
PARALLEL_SIZE = 16
EVAL_NUM = 50

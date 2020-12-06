from os import environ


try:
    LOCAL_PATH = environ['LOCAL_PATH']
except:
    LOCAL_PATH = '/home/juypter/trinity/pandas-polygon/data'

try:
    S3_PATH = environ['S3_PATH']
except:
    S3_PATH = 'polygon-equities/data'

try:
    POLYGON_API_KEY = environ['POLYGON_API_KEY']
except:
    POLYGON_API_KEY = 'seceret'

try:
    ALPHAVANTAGE_API_KEY = environ['ALPHAVANTAGE_API_KEY']
except:
    ALPHAVANTAGE_API_KEY = 'seceret'

try:
    TIINGO_API_KEY = environ['TIINGO_API_KEY']
except:
    TIINGO_API_KEY = 'seceret'

try:
    B2_ACCESS_KEY_ID = environ['B2_ACCESS_KEY_ID']
except:
    B2_ACCESS_KEY_ID = 'seceret'

try:
    B2_SECRET_ACCESS_KEY = environ['B2_SECRET_ACCESS_KEY']
except:
    B2_SECRET_ACCESS_KEY = 'seceret'

try:
    B2_ENDPOINT_URL = environ['B2_ENDPOINT_URL']
except:
    B2_ENDPOINT_URL = 'seceret'

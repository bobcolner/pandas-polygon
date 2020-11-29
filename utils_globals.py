from os import environ


try:
	LOCAL_PATH = environ['LOCAL_PATH']
	S3_PATH = environ['S3_PATH']
except:
	LOCAL_PATH = '/home/juypter/trinity/pandas-polygon/data'
	S3_PATH = 'polygon-equities/data'

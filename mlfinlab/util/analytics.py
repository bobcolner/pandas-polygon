# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module allows us to track how the library is used and measure statistics such as growth and lifetime.
"""


# Analytics
import os
import hashlib
from requests import get

import analytics

# Don't run inside travis
IS_TRAVIS = False
# pylint: disable=bare-except
try:
    IS_TRAVIS = bool(os.environ['IS_TRAVIS'])
except:
    pass


# Set Location and User based on IP
IP = None
USER = "unknown"

# pylint: disable=bare-except
try:
    IP = get('https://api.ipify.org').text
    USER = hashlib.md5(IP.encode()).hexdigest()
except:
    pass

# Connect with DB
analytics.write_key = 'vAOWsGr5lqPmMMMevTCzowpTuzIFylgd'

# Set location
if IP:
    LOCATION = {"ip": IP}
else:
    LOCATION = None


# Generic function for pinging the server
def page(url):
    """
    Generic function used to send information back to the server.
    :param url: function used.
    """
    if not IS_TRAVIS:
        analytics.page(USER, 'Docs', 'Python',
                       {"url": url},
                       context=LOCATION)

"""Utility functions to help with logging in various places throughout repo."""
import os
import requests
import logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)
# Slack setup
HOOK_URL = None
SLACK_CHANNEL = None


def _send_slack_status_log_print(text, channel=None, level=logging.INFO, send_status_update=True, v=True):
    if send_status_update:
        _send_slack_msg(text, channel)
        _send_status(text)
    _log(text, level=level, v=v)


def _send_slack_msg(text, channel=None):
    """
    """
    try:
        global HOOK_URL, SLACK_CHANNEL
        if HOOK_URL is None:
            HOOK_URL = os.getenv('SlackHookUrl')
            lg.info('HOOK_URL: %s' % HOOK_URL)
        if SLACK_CHANNEL is None:
            SLACK_CHANNEL = os.getenv('slackChannel')
            lg.info('SLACK_CHANNEL: %s' % SLACK_CHANNEL)

        if channel is None:
            channel = SLACK_CHANNEL
        slack_msg = {
            'channel': channel,
            'text': text
        }
        r = requests.post(HOOK_URL, json=slack_msg)
        if r.status_code is not requests.codes.ok:
            print(r.status_code)
            print(r.reason)
            print(r.text)
    except Exception as e:
        _log(str(e))

def _send_status(status):
    """
    When we get around to implementing this, make sure to wrap it in a try/except block so that it
    doesn't crash the system if there is an communication error.
    """
    pass


def _log(text, level=logging.INFO, v=True):
    if v:
        print(text)
    if level == logging.DEBUG:
        lg.debug(text)
    elif level == logging.INFO:
        lg.info(text)
    elif level == logging.WARNING:
        lg.warning(text)
    elif level == logging.ERROR:
        lg.error(text)
    else:
        lg.warning('Unrecognized logging level (%s) associate with log message %s' % (str(level), str(text)))


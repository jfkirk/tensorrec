"""

    Session Management
    ~~~~~~~~~~~~~~~~~~

"""
import tensorflow as tf

_session = None


def get_session():
    """

    Returns:

    """
    global _session

    # Build/retrieve the session if it doesn't exist
    if _session is None:
        if tf.get_default_session() is not None:
            _session = tf.get_default_session()
        else:
            _session = tf.Session()

    return _session


def set_session(session):
    """

    Args:
        session:

    Returns:

    """
    global _session
    _session = session

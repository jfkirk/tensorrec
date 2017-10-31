# Suppress TensorFlow logging when testing
import logging
logging.getLogger('tensorflow').disabled = True

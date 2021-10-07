import logging

cool_mc_logger = logging.getLogger('cool_mc_logger')
cool_mc_logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('./logs.log')
fh.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(project)s - %(task)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to logger
cool_mc_logger.addHandler(fh)

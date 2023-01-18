from autoIG.instruments import Epics

model_name = "knn-reg-model"
model_version = "27"

r_threshold = 0.9  # threshold to buy
# To get the price stream of, must match model used. TODO: check this using param set in model with mlflow
# epic =  Epics.US_CRUDE_OIL.value  # market only open to trading until 10pm
# close_after_x_mins = 1 # How long to hold a position for
# stream_length_needed = 3 # How long we need the stream to be before we start TODO: Make only recent streams count

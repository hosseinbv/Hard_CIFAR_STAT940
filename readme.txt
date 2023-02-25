#====================================================================================================================
# Note: In this work, I used the idea of sharpness aware minimizer (SAM), which costs double in computing the gradient and space complexity but worked well in this data challenge.
# SAM uses the first gradient to search a locality of weights and takes a direction that minimizes the worst case in that locality.
# Please see this ref for more details: https://arxiv.org/abs/2102.11600
# I also used the ASAM code in this github for ASAM minimizer: https://github.com/SamsungLabs/ASAM.git
# ===================================================================================================================
#  ========================================== My Best Acc on Kaggle leaderboard is 0.8266


    ## Please note that I used a semi-supervised approach for training. That is, after several epoches, I make a prediction over test data (no label data), and 
    ## accept the predicted labels for those which come with a high prediction logit scores and add them into my training data. I do this startegy for the following epoches as well.
    ## To do that, I made two optimizers/scheduler/minimizer, defining as follwos:


To run the code, first make an env and run: pip install -r requirements.txt
then, run train.py in the same env.
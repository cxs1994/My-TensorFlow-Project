# We are going to solve two problems here
  1. How to do training and validataion at the same time with Tensorflow? (more details about this question: [here](http://stackoverflow.com/questions/41162955/tensorflow-queues-switching-between-train-and-validation-data))
  2. How to plot training and validation curvers on Tensorboard?
  
# The method (details are in the CODE)
  1. generate train and validation batch with two queues
  2. fetch the contents of each queue independently with sess.run()
  3. during training, use feed_dict to select which one to be pushed into the computational graph
  
# help:
  http://www.cnblogs.com/cxscode/p/8476966.html

function [w, w0] = process_epoch(w, w0, lrate, examples, labels)

[wgrad, w0grad] = partial_gradient(w,w0,examples,labels);

w=w-lrate*wgrad;
w0=w0-lrate*w0grad;
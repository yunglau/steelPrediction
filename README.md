# How to make a money printing machine in one night

We implemented for the ImplementAI 2019 event 2 predictive algorithm for steel prices.
Those projet were suggested by Hatch company, which provided important insights on the steel markets.

We used LSTM based neural network and ODEnet (Chen, R.~T.~Q., Rubanova, Y., Bettencourt, J., et al. 2018, arXiv e-prints, arXiv:1806.07366). To the best of our knowledge, later approach was never used in commidity prediction.
The Odenet implementation takes great advantage on Chen's implementation, which can be found here: https://github.com/rtqichen/torchdiffeq.

Project objectives
  
  The goal was to create model able to predict steel prices evolution over on long time range.
  As have demonstrated our first observations, standard linear models for such financial evolutions are poorly performing.
  Due to time constraints we focus on gathering 9 main analytics and built too neural network predictive model out of them.
    
Results
    
  We achieved short range prediction with good accuracy but our model fall short of prediction capability quite fast.
  Time was spend on gathering open data that can be found in our repository, building the neural network and training them.
    
Work possible flows:

  While this work initiates some novel approach in tackling financial noisy time series data, it does lack a second analysis that we would have loved to put in place. Indeed it would have been interesting to look at the effect of sudden important variation in one of the feature. Our prediction is that the Odenet, approximating in a sense the real network dynamic, could highlights interesting analytical effects to this regard. 
    
Features technical details


Model technical details
     
   LSTM
     
     
     
   ODEnet based model:
The network is composed of three network structured similarly than an autoencoder.
A first network, the encoder, is in charge of smmarizing the past information in a condensed representation.
This reprenzation is used by an ODEnet. This neural network approximate the derivative of its input with respect to the time. Using this derivative an ODEsolver builds, starting from the condensed representation, a new reprensation for every input time. It then extend it prediction to unseen data. Finally a decoder, a simple feed-forward neural network, moves these representation back to their initial representation spaces.
The training loss is obtained from the "Neural Ordinary Differential Equations" paper repository. At prediction time we only analysed prediction for the steel price (due to lack of times).
Optimisation scheme is Adam, also inspired from "Neural Ordinary Differential Equation"
   










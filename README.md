# How to make a money printing machine in one night.

We implemented for the ImplementAI 2019 event (24h) 2 predictive algorithm for steel prices.
Those projet were suggested by Hatch company, which provided important insights on the steel markets and idea on how to approach the problem.

We used LSTM based neural network and ODEnet (Chen, R.~T.~Q., Rubanova, Y., Bettencourt, J., et al. 2018, arXiv e-prints, arXiv:1806.07366). To the best of our knowledge, latter approach was never used in commodity prediction.
The Odenet implementation takes great advantage on Chen's implementation, which can be found here: https://github.com/rtqichen/torchdiffeq.

## Project objectives
  
  The goal was to create model able to predict steel prices evolution over on long time range.
  As have demonstrated our first observations, standard linear models for such financial evolutions are poorly performing.
  Due to time constraints we focus on gathering 9 main analytics and built too neural network predictive model out of them.
  
  Despite the relative narrow range of the model application, there is powerful extension one can make out of the proof of concept of using ODEnet for predictions on complex dynamics.
Among possible ideas:
  - prediction of behaviors of set of molecular reactions (molecular engineering).
  - new approach to climate modeling.
  - application to other markets.
    
## Results
    
  We achieved short range prediction with good accuracy but our model fall short of prediction capability quite fast.
  Time was spend on gathering open data that can be found in our repository, building the neural network and training them.
    
## Work possible flows:

  While this work initiates some novel approach in tackling financial noisy time series data, it does lack a second analysis that we would have loved to put in place. Indeed it would have been interesting to look at the effect of sudden important variation in one of the feature. Our prediction is that the Odenet, approximating in a sense the real network dynamic, could highlights interesting analytical effects to this regard. 
   
  We also raise the concern that different approach could have been adopted. Network architecture as: Boltzmann machines are particularly efficient for reprensint interactions between multiple time series (here our selected features). Such tool were introduced a by Hinton : http://www.scholarpedia.org/article/Boltzmann_machine, and could have been of relevancy in such project.
    
## Features technical details

  The feature used were:
    -
    -
    -
    -
    -
    

## Model technical details
     
   LSTM
     
![](Images/ODEnet.jpg?raw=true "Title")
     
   ODEnet based model:
The network is composed of three network structured similarly than an autoencoder.
A first network, the encoder, is in charge of smmarizing the past information in a condensed representation.
This reprenzation is used by an ODEnet. This neural network approximate the derivative of its input with respect to the time. Using this derivative an ODEsolver builds, starting from the condensed representation, a new reprensation for every input time. It then extend it prediction to unseen data. Finally a decoder, a simple feed-forward neural network, moves these representation back to their initial representation spaces.
The training loss is obtained from the "Neural Ordinary Differential Equations" paper repository. At prediction time we only analysed prediction for the steel price (due to lack of times).
Optimisation scheme is Adam, also inspired from "Neural Ordinary Differential Equation"
   

Team: Elaine Lau, Ying Yang, Hector Perez Hoffman, Pierre Orhan

We would like to thank the ImplementAI2019 team, as well as sponsors: Hatch, Coveo, NBC, Wrnch, Samasource,CEA.






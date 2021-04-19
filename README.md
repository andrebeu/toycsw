# main: handheld splitting + schema rnn
- splitting controlled by RNN
  - 04/19/21: found test B>I without much optimizing.
- next steps:
  - optimize learnrate and stsize for better initial learning 
  - gridsearch splitting params to assess robustness and look for best fit
  - change eval method to normalized 2afc


## delta+rnn branch
- splitting controled by delta learner, performance evaluated on rnn
  - good B>I without much optimizing

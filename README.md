# main: handheld splitting + schema rnn
- splitting controlled by RNN
  - 04/19/21: found test B>I without much optimizing.

### working notes:
  - need to optimize stsize
    - fit initial trials of blocked 
    - set to default
  - then, gridsearching splitting params
    - gs script is ready to run


## delta+rnn branch
- splitting controled by delta learner, performance evaluated on rnn
  - good B>I without much optimizing

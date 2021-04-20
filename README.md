# main: best fit
- splitting controlled by RNN
  - 04/19/21: found test B>I without much optimizing.
  - freezing this branch with results from current bestfit B>I
  - {'sticky_decay': 0.02, 'pe_thresh': 0.9, 'init_lr': 0.45, 'lr_decay': 0.25, 'stsize': 6.0}


### working notes:
  - analysis of best fit handheld splitting + schema RNN
    - plot splitting
    - best fit: 
      - I more splits 
      - I never enters pe`<`thresh
      - B more pr_stay
  - can I simplify splitting procedure?


### notes
  - best fit: {'sticky_decay': 0.02, 'pe_thresh': 0.9, 'init_lr': 0.45, 'lr_decay': 0.25, 'stsize': 6.0}
  - stsize 10 yields poor fit (I learns too fast)

## delta+rnn branch
- splitting controled by delta learner, performance evaluated on rnn
  - good B>I without much optimizing

# p4

To run this, just use the stub file, all of our code is there. The stub_final is also set up to perform 100 iterations of 100 games and will record the stats of every 100 games.

For instance, from within the code folder, just do:

  python stub_final.py
  
Also note, for reporting purposes (we wanted to see how we were dying), we added a variable internal to swingymonkey called 'self.death' that would report how the monkey died. We just did this out of curiousity and it does not enhance actual runtime performance. If you run our stub file together with the original swingymonkey it raises are error because we call this death value. We aren't sure if you plan on doing this... 

Other files of interest in this repo:
stub_one_param_tuning - used to tune one parameter at a time, does multipled iterations with each param value
stub_mult_param_tuning - used to tune a grid of params

SwingyMonkey_no_render - version of file with all rendering commented out

# How to use generate_trials_*.py

Configure utils/generate_eval_trial_sets.py to generate the trial xml config files.  
    
* The script generates a set of config files that describes a trial. Each trial starts with non-novel episodes, and at some point transitions to novel episodes
    * Set the number of total levels using `NUM_LEVELS`
    * Set the episode at which novelty is introduced using `LEVELS_BEFORE_NOVELTY` (if you are looking to make a 100% novel trial, set this value to 0.)
* Note the `NON_NOVEL_TO_USE` dictionary - all non novel levels / types will be used to generate the set of config files.  
* Note the `NOVEL_TO_USE dictionary` - all novel levels / types will be used to generate the set of config files. A config file will be generated for every non-novel - novel combination. 
* Use the `REPETITION` setting to set how many times a level will be repeated within a trial (ie, a trial with `NUM_LEVELS = 5`, `LEVELS_BEFORE_NOVELTY = 1` and `REPETITION = 2` will result in `[non_novel_a, novel_b, novel_b, novel_c, novel_c]`)
You can execute the script by using `python3 utils/generate_trials_<insert domain here>.py`. This should output new config xml files to the `/Phase2` directory

## Additional notes:
The `NOVEL_TO_USE` and `NON_NOVEL_TO_USE` dictionaries are pools to sample levels/types from. However, ultimately the parameter that controls whether or not to generate non-novelty is the `LEVELS_BEFORE_NOVELTY` parameter. If this is set to 0, then all levels within the trial will be novel. Config files will still be generated for each non-novel/novel pair, but they will not contain any non-novel levels due to `LEVELS_BEFORE_NOVELTY = 0`
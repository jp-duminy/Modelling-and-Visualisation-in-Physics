# Modelling and Visualisation in Physics 2026

Welcome to my repository for the course PHYS10035, Modelling and Visualisation in Physics, taken in 2026. This course focuses on developing core simulation techniques, including Monte Carlo methods, cellular automata, and partial differential equations; furthermore, it integrates visualisation techniques and user interactivity. There is an emphasis on writing efficient, flexible code which can be readily used and understood by others.

There were three checkpoints, each testing another core technique:

- CP1: the Ising Model with Monte Carlo methods.
- CP2: the Game of Life and SIRS cellular automata.
- CP3: solving the Cahn-Hilliard and Poisson equations.

Followed by an exam which puts a twist on one of the checkpoints. In my case, the exam modified CP1.

## Contents

There are three checkpoint folders, each containing the coursework in the state I submitted it. My grades were:

- [CP1](checkpoint-1): 14/15 (took the absolute of the mean magnetisation, should be other way around)
- [CP2](checkpoint-2): 15/15
- [CP3](checkpoint-3): 18.5/20 (incorrect BCs on magnetic field wire, wire should be periodic)

Potentially of more interest would be the folder [exam](exam). I rewrote my checkpoint code to be more efficient, modular, and agnostic in preparation for the exam. These files thus contain faster, better code which also has extensions (e.g. Potts model as well as Ising for CP1). 

There are then folders corresponding to my solutions for past papers; the 2026 folder (to be uploaded August) contains the code I wrote on the spot for the in-person exam, though I have omitted my text answers and datafiles as I suppose they count as exam scripts which should not be shared. The prevailing sentiment in my cohort was the exam was notably difficult; to any future students, I would advise doing what I did and modifying your checkpoint code to be agnostic, as this let me quickly adapt it in the exam and comfortably finish all questions. Doing so will also make you more comfortable with the physics behind the code, lending you the flexibility to understand how it should be adapted for an unseen problem; many people in my cohort struggled to make it past the first question of the exam. 

Though I am an astrophysicist and MVP leans more towards matter physics, I still thoroughly enjoyed the course and think it punches above its weight in terms of the value it provides for the ten credits it requires; it is well-taught and its time investment is proportional. I would strongly recommend it, even if coding is not your strong suit. 


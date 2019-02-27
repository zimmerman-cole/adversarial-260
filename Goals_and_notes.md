# Coding tasks

* Create target MNIST network that works with Foolbox library
  * Can either create our own from scratch
    * Pro: more flexibility
    * Con: have to be careful about getting it to work with Foolbox
  * Or, can use pre-trained model
    * Pro: simple, code already written and working in blackbox_attack.ipynb, already has decent accuracy
    * Con: less flexibility in target architecture
* Implement PGD/multi-step FGSM attack
  * If target network similar to example one in blackbox_attack.ipynb, then should just be a matter of copying and pasting blackbox_attack code and modifying a few lines.

# Project flow
1. Split test data into sets Test 1 (70%) and Test 2 (30%)
  * This will be important when creating and testing adversarial data
2. Train target on training data (or just use pre-trained model)
3. Get target's accuracy on training data
  * Motivation: create a baseline of performance
4. Get target's accuracy on Test 1
  * Motivation: create a baseline of how network performs on unseen data
5. Generate white-box adversarials from Test 1, call it set Test 1-Adv
6. Get target's accuracy on Test 1-Adv
  * Motivation: "How effective is the white-box attack using previously unseen data?"
7. Train target on Test 1-Adv, call it new-target
  * This is the defense aspect of the project
8. Get new-target's accuracy on Test 1-Adv
  * Motivation: "How effective was defensive training in increasing robustness to an identical attack in the future (i.e. using exact same adversarials)?"
9. Generate white-box adversarials from Test 2, call it set Test 2-Adv
10. Get accuracy of new-target on Test 2-Adv
  * Motivation: "How effective was defensive training in increasing robustness to an attack of same type (FGSM), but using unseen data (i.e. using adversarials not used in defensive training)?"
11. Get accuracy of new-target on Test 1 and Test 2 (non-adv)
  * Motivation: "Did defensive training compromise our accuracy on non-adversarial examples?"
12. Generate black-box adversarials from Test 2, call it Test 2-bb
13. Get accuracy of target on Test 2-bb
  * Motivation: "How effective is a black-box attack on the network?"
14. Get accuracy of new-target on Test 2-bb
  * Motivation: "How effective was defensive training on one type of attack in increasing robustness to an unseen type of attack?"

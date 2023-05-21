# CrossSplit: Mitigating Label Noise Memorization through Data Splitting
PyTorch Code for the paper "CrossSplit: Mitigating Label Noise Memorization through Data Splitting" (ICML 2023)
![fig_arch7](https://user-images.githubusercontent.com/100881552/235355178-d426d9e1-30e8-40a5-a281-502edb31c254.png)

Run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5 

Run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --noise_mode 'sym' --r 0.9 

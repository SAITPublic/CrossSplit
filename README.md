# CrossSplit: Mitigating Label Noise Memorization through Data Splitting
Official PyTorch Code for the ICML 2023 paper "CrossSplit: Mitigating Label Noise Memorization through Data Splitting" https://proceedings.mlr.press/v202/kim23a.html

![fig_arch7](https://user-images.githubusercontent.com/100881552/235355178-d426d9e1-30e8-40a5-a281-502edb31c254.png)

Run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5 

Run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --noise_mode 'sym' --r 0.9 

 # Citation
 	@InProceedings{Jihye_2023_ICML,
      title={CrossSplit: Mitigating Label Noise Memorization through Data Splitting}, 
      author={Jihye Kim and Aristide Baratin and Yan Zhang and Simon Lacoste-Julien},
      booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
      year={2023}, 
      pages = {16377-16392}
      }

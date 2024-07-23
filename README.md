# CarFormer: Self-Driving with Learned Object-Centric Representations


The choice of representation plays a key role in self-driving. Bird’s eye view (BEV) representations have shown remarkable performance in recent years. In this paper, we propose to learn object-centric representations in BEV to distill a complex scene into more actionable information for self-driving. We first learn to place objects into slots with a slot attention model on BEV sequences. Based on these object-centric representations, we then train a transformer to learn to drive as well as reason about the future of other vehicles. We found that object-centric slot representations outperform both scene-level and object-level approaches that use the exact attributes of objects. Slot representations naturally incorporate information about objects from their spatial and temporal context such as position, heading, and speed without explicitly providing it. Our model with slots achieves an increased completion rate of the provided routes and, consequently, a higher driving score, with a lower variance across multiple runs, affirming slots as a reliable alternaive in object-centric approaches. Additionally, we validate our model’s performance as a world model through forecasting experiments, demonstrating its capability to accurately predict future slot representations.

<img width="800" alt="CarFormer overview" src="assets/carformer-overview.png">

---

## 🚗 Demo

https://github.com/user-attachments/assets/2cd76957-fc6b-4a35-bc14-500708a767a1



## Evaluation Results 

### Closed-Loop Evaluation on Longest6
<img width="800" alt="Longest 6 results" src="assets/results-longest6.png">

### Forecasting BEV 
<img width="800" alt="Forecasting results" src="assets/results-forecasting.png">

## 🗓️ TODOs

- [ ] Release code for training (Slots, Attributes)
- [ ] Release code for agent/evaluation (Slots, Attributes)
- [ ] Release pretrained checkpoints

## 🎫 License

This project is released under the [MIT license](LICENSE). 

## 🖊️ Citation

If you find this project useful in your research, or if you reuse the code for other purposes, please consider citing:

```BibTeX
@misc{hamdan2024carformer,
      title={CarFormer: Self-Driving with Learned Object-Centric Representations}, 
      author={Shadi Hamdan and Fatma Güney},
      year={2024},
      eprint={2407.15843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15843}, 
}
```

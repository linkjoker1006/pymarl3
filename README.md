# Environment configuration
Please refer to the environment configuration https://github.com/tjuHaoXiaotian/pymarl3  

# Train and test
The training command is in train.sh, and the testing command is in test.sh. It includes the parameters used by various algorithms in each scenario.  
## Additional note during testing
If you want to observe the model's performance, please uncomment the corresponding content under # test and # test with chunksize in parallel_runner.py. If you want to observe the AMR of the model, please remove the annotation for the corresponding content under # AMR 统计.  

# Notes
The code has been updated to the final version, and no further research will be done after that. Please email me if there are any issues with the operation or results. Due to busy work schedules, we do not have the energy to maintain this project to perfection. However, we can guarantee that although there may be some minor issues during the final stage of the organization, the entire project is usable, and the performance described in the article can be replicated.  

# Citing QTypeMix
```tex
@article{fu2025qtypemix,
   title={QTypeMix: Enhancing multi-agent cooperative strategies through heterogeneous and homogeneous value decomposition},
   author={Fu, Songchen and Zhao, Shaojing and Li, Ta and Yan, Yonghong},
   journal={Neural Networks},
   volume={184},
   pages={107093},
   year={2025},
   publisher={Elsevier}
}
```

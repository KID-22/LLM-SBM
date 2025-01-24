The [dataset_info.json](dataset_info.json) contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

Our experienments utilize datasets in **sharegpt** format.
Specifically, we added the `weight` feature based on LLaMA-Factory, so the data format should be:

```json
{
    "conversations": [
        {
            "from": ,
            "value": 
        },
        {
            "from": ,
            "value": 
        }
    ],
    "chosen": {
        "from": ,
        "value": 
    },
    "rejected": {
        "from": ,
        "value": 
    },
    "weight": 
},
```
See [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) for more details.





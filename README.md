# SpanSRL
## Setting
### BERT
* Fine-tuning : all
* epoch : 30
* patience : 5
* rate : BERT=5e-5, Linear=1e-4

### Rinna / CyberAgent (GPTNeoX)
* Fine-tuning : LoRA (layer : qvk / r=8 / )
* epoch : 30
* patience : 3
* rate : 1e-3 (3) -> 2e-4 (after 3)
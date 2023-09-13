# Large-scale Dataset and Effective Model for Variant-Disease Associations Extraction

This repository contains the code and data for the paper "Large-scale Dataset and Effective Model for Variant-Disease Associations Extraction" accepted by ACM-BCB 2023.

## VDAL

VDAL is the first constructed large-scale dataset for document-level VDA extraction task. Specifically, the basis of VDAL (i.e., the variant-disease association pair truth labels and the trusted PubMed ID source information) are from the DisGetNet, and we use the PubTator tool to access the variant and disease entity annotations of the corresponding PubMed documents, together to construct VDAL for addressing the document-level VDA extraction problem.

## VDANet

VDANet is a simple yet and effective model for addressing the document-level VDA extraction task as a powerful baseline for follow-up studies. The only difference between VDANet and other baseline models (e.g., BioBERT) is the design of the embedding layer. General models add the token embeddings, position embeddings and segment embeddings together in the embedding layer as the initial input representation, while VDANet adds another corresponding gene embeddings of the variants into the embedding layer, which serve as a bridge for exploring the associations between genetic variants and diseases more effectively, tailored for the document-level VDA extraction task.

## Requirements

```
python>=3.6
pytorch==1.10.2
transformers==4.18.0
numpy==1.19.5
```

## Quick Start

Put the VDAL dataset (including `vdal_train.data`, `vdal_dev.data` and `vdal_test.data`) in folder `./dataset/vda`.

### Train `VDANet`

```
python train_vda.py
```
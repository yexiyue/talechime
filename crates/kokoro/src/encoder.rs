//! Official ALBERT sharing and text/prosody encoders.
use crate::ops::{Weights, affine_norm, leaky, normalize};
use candle_core::{DType, Result, Tensor};

impl Weights {
    pub fn albert(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, length) = ids.dims2()?;
        let words = self
            .get("bert.embeddings.word_embeddings.weight")?
            .index_select(&ids.flatten_all()?, 0)?
            .unsqueeze(0)?;
        let positions = self
            .get("bert.embeddings.position_embeddings.weight")?
            .narrow(0, 0, length)?
            .unsqueeze(0)?;
        let types = self
            .get("bert.embeddings.token_type_embeddings.weight")?
            .narrow(0, 0, 1)?
            .unsqueeze(0)?;
        let embeddings = (words + positions)?.broadcast_add(&types)?;
        let embeddings = self.layer_norm(&embeddings, "bert.embeddings.LayerNorm", 1e-12)?;
        let mut x = self.linear(&embeddings, "bert.encoder.embedding_hidden_mapping_in")?;
        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        for _ in 0..12 {
            let q = self
                .linear(&x, &format!("{p}.attention.query"))?
                .reshape((1, length, 12, 64))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = self
                .linear(&x, &format!("{p}.attention.key"))?
                .reshape((1, length, 12, 64))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = self
                .linear(&x, &format!("{p}.attention.value"))?
                .reshape((1, length, 12, 64))?
                .transpose(1, 2)?
                .contiguous()?;
            let context = crate::attention::forward(&q, &k, &v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, length, 768))?;
            let attention = self.linear(&context, &format!("{p}.attention.dense"))?;
            x = self.layer_norm(
                &(x + attention)?,
                &format!("{p}.attention.LayerNorm"),
                1e-12,
            )?;
            let ffn = self.linear(&x, &format!("{p}.ffn"))?.gelu()?;
            let ffn = self.linear(&ffn, &format!("{p}.ffn_output"))?;
            x = self.layer_norm(&(x + ffn)?, &format!("{p}.full_layer_layer_norm"), 1e-12)?;
        }
        Ok(x)
    }

    pub fn duration_encoder(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let s = style.unsqueeze(1)?.broadcast_as((b, t, 128))?;
        let mut x = Tensor::cat(&[x, &s], 2)?;
        for layer in 0..3 {
            x = self.bilstm(&x, &format!("predictor.text_encoder.lstms.{}", layer * 2))?;
            let params = self
                .linear(
                    style,
                    &format!("predictor.text_encoder.lstms.{}.fc", layer * 2 + 1),
                )?
                .unsqueeze(1)?;
            x = normalize(&x, 2, 1e-5)?
                .broadcast_mul(&(params.narrow(2, 0, 512)? + 1.0)?)?
                .broadcast_add(&params.narrow(2, 512, 512)?)?;
            x = Tensor::cat(&[&x, &s], 2)?;
        }
        Ok(x)
    }

    pub fn text_encode(&self, ids: &Tensor) -> Result<Tensor> {
        let mut x = self
            .get("text_encoder.embedding.weight")?
            .index_select(&ids.flatten_all()?, 0)?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        for layer in 0..3 {
            x = self.conv(&x, &format!("text_encoder.cnn.{layer}.0"), 1, 1)?;
            let p = format!("text_encoder.cnn.{layer}.1");
            x = affine_norm(
                &x.transpose(1, 2)?,
                self.get(&format!("{p}.gamma"))?,
                self.get(&format!("{p}.beta"))?,
                1e-5,
            )?;
            x = leaky(&x, 0.2)?.transpose(1, 2)?;
        }
        self.bilstm(&x.transpose(1, 2)?, "text_encoder.lstm")?
            .transpose(1, 2)?
            .to_dtype(DType::F32)
    }
}

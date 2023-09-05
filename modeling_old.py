import torch
from get_model_new import get_model, get_model_config
import pytorch_lightning as pl


class Encoder(torch.nn.Module):
    def __init__(self, model_name, version='B', layer=12, device='cuda', img_size=224):
        super().__init__()
        self.model_name = model_name
        self.model_config = get_model_config(model_name, version)
        self.model = get_model(model_name, version, device, img_size)
        if layer != 12:
            self.num_layers = layer
        else:
            self.num_layers = self.model_config['layers']

    def forward(self, x):
        forward_function = getattr(self, f'forward_{self.model_name}')
        return forward_function(x)

    def forward_dino(self, x):
        x = self.model.prepare_tokens(x)
        for blk in self.model.blocks[:self.num_layers]:
            x = blk(x)
        x = self.model.norm(x)

        return x[:, 1:, :]

    def forward_dinov2(self, x):
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks[:self.num_layers]:
            x = blk(x)
        x = self.model.norm(x)

        return x[:, 1:, :] 

    def forward_sup_vit(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        for layer in self.model.encoder.layers[:self.num_layers]:
            x = layer(x)

        x = self.model.encoder.ln(x)
        # x = self.model.encoder(x)

        return x[:,1:]

    def forward_mae(self, x):
        img_enc, _, _ = self.model.forward_encoder(x, mask_ratio=0, layer=self.num_layers)

        return img_enc[:, 1:]
    
    def forward_simmim(self, x):
        x = self.model.encoder.patch_embed(x)

        batch_size, seq_len, _ = x.size()

        cls_tokens = self.model.encoder.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.model.encoder.pos_embed is not None:
            x = x + self.pos_embed
        x = self.model.encoder.pos_drop(x)

        rel_pos_bias = self.model.encoder.rel_pos_bias() if self.model.encoder.rel_pos_bias is not None else None
        for blk in self.model.encoder.blocks[:self.num_layers]:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.model.encoder.norm(x)
        if self.model.encoder.fc_norm is not None:
            t = x[:, 1:, :]
            return self.model.encoder.fc_norm(t.mean(1))
        else:
            return x[:, 1:]



class Decoder(torch.nn.Module):
    def __init__(self, num_classes, emb_size=768, classifier_name='linear') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.classifier_name = classifier_name

        if classifier_name == 'linear':
            self.linear = torch.nn.Linear(emb_size, num_classes+1)
            self.head = self._linear_classifier

        elif classifier_name == 'linear_mae':
            self.linear = torch.nn.Sequential(torch.nn.BatchNorm1d(emb_size, affine=False, eps=1e-6),
                                            torch.nn.Linear(emb_size, num_classes+1))
            self.head = self._linear_classifier

        elif classifier_name == 'upsample':
            self.bilinear = torch.nn.Upsample(scale_factor=(4, 4), mode='bilinear', align_corners=False)
            self.linear = torch.nn.Linear(emb_size, num_classes+1)
            self.head = self._upsample
        
        elif classifier_name == 'deconvolution':
            self.convtranspose2d = torch.nn.ConvTranspose2d(in_channels=emb_size, out_channels=num_classes+1,
                                kernel_size=(4, 4), stride=4)
            self.head = self._deconv
        
        else:
            raise ValueError('Training with bilinear upsampling should be with frozen backbone')

    def forward(self, img_enc):
        img_enc = self.head(img_enc)

        return img_enc

    def _upsample(self, img_enc):
        # img_enc = img_enc.reshape(-1, 64, 128, img_enc.shape[-1])
        img_enc = img_enc.reshape(-1, 14, 14, img_enc.shape[-1])
        img_enc = img_enc.permute(0, 3, 1, 2)
        img_enc = self.bilinear(img_enc)
        img_enc = img_enc.permute(0, 2, 3, 1)
        img_enc = self.linear(img_enc)
        img_enc = img_enc.reshape(img_enc.shape[0], -1, img_enc.shape[-1])
        # img_enc = self.activation(img_enc)

        return img_enc

    def _deconv(self, img_enc):
        print("before reshape:", img_enc.shape)
        img_enc = img_enc.reshape(-1, 14, 14, img_enc.shape[-1]) # TODO add reshape as layer
        print("after reshape:", img_enc.shape)

        img_enc = img_enc.permute(0, 3, 1, 2)
        img_enc = self.convtranspose2d(img_enc)
        img_enc = img_enc.permute(0, 2, 3, 1)
        # img_enc = self.activation(img_enc)
        img_enc = img_enc.reshape(img_enc.shape[0], -1, img_enc.shape[-1])

        return img_enc

    def _linear_classifier(self, img_enc):
        img_enc = self.linear(img_enc)
        # img_enc = self.activation(img_enc)

        return img_enc


class Segmenter(pl.LightningModule):
    def __init__(self, backbone_name, num_classes, emb_size=768,  classifier_name='linear', backbone_freeze=False, read_embeds=False) -> None:
        super().__init__()
        if classifier_name == 'upsample' and not backbone_freeze:
            raise ValueError('Training with upsampling should be with frozen backbone')

        if classifier_name == 'deconvolution' and not backbone_freeze:
            raise ValueError('Training with deconvolution should be with frozen backbone')

        self.backbone_freeze = backbone_freeze
        self.encoder = None
        model_config = get_model_config(backbone_name)
        # emb_size = model_config['emb_size']
        print('##################')
        print(emb_size)
        print('##################')
        if not read_embeds:
            self.encoder = Encoder(backbone_name)
            if self.backbone_freeze:
                self.encoder.requires_grad_(False)
        self.decoder = Decoder(num_classes, emb_size, classifier_name)

    def forward(self, x):
        if self.encoder:
            if self.backbone_freeze:
                self.encoder.eval()
                with torch.no_grad():
                    x = self.encoder.forward(x)
            
            else:
                x = self.encoder.forward(x)
            
        x = self.decoder(x)

        return x
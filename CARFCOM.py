class BasicConv1(nn.Module):
    def __init__(self, 
                 c_in,
                 heads):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size = 1, stride = 1, padding = 0, groups=heads),
            nn.BatchNorm2d(c_in),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

class BasicConv2(nn.Module):
    def __init__(self, 
                 c_in,
                 r = 4):
        super().__init__()
        c_inter = int(c_in // r)
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_inter, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(c_inter),
            nn.SiLU(),
            nn.Conv2d(c_inter, c_in, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(c_in),
            nn.SiLU(),
        )        
    def forward(self, x):
        return self.block(x)

class BasicConv3(nn.Module):
    def __init__(self, 
                 c_in):
        super().__init__()
        c_out = int(c_in // 2)
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(c_out),
            nn.SiLU()
        )        
    def forward(self, x):
        return self.block(x)

class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()        
        self.lambd = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        lambd = self.sigmoid(self.lambd)
        out = x1 * lambd + x2 * (1 - lambd)
        return out

class MyMultiAttention(nn.Module):
    def __init__(self, 
                 c_in,
                 n = 5,
                 spectrum = None):
        super().__init__()
        self.heads = 4
        assert c_in % self.heads == 0, f"通道数必须可以被heads整除,当前heads为{self.heads}"
        self.head_dim = c_in // self.heads
        self.n = n
        self.pad = n // 2
        self.spectrum = spectrum
        
        # 初始化每个头的卷积层
        if spectrum == 'ir':
            self.ir_q = BasicConv1(c_in, self.heads)
            self.ir_k = BasicConv1(c_in, self.heads)
            self.ir_v = BasicConv1(c_in, self.heads)
            self.gamma1 = nn.Parameter(torch.ones(1, self.heads, 1, 1))
        elif spectrum == 'rgb':
            self.rgb_q = BasicConv1(c_in, self.heads)
            self.rgb_k = BasicConv1(c_in, self.heads)
            self.rgb_v = BasicConv1(c_in, self.heads)
            self.gamma2 = nn.Parameter(torch.ones(1, self.heads, 1, 1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_k(self, tensor):
        bs, c, h, w = tensor.shape
        return tensor.view(bs, c, h*w, 1, 1).permute(0, 2, 1, 3, 4).contiguous()  # 合并维度在前

    def pad_tensor(self, tensor):
        bs, c, h, w = tensor.shape        
        # 左右填充 (直接拼接列)
        left = tensor[..., self.pad+1 : self.pad+1+self.pad]       # [bs,c,h,pad]
        right_start = w - 2*self.pad - 1
        right = tensor[..., right_start : right_start+self.pad]    # [bs,c,h,pad]
        padded = torch.cat([left, tensor, right], dim=3)           # [bs,c,h,w+2pad]
        
        # 上下填充 (基于左右填充后的结果拼接行)
        upper = padded[:, :, self.pad+1 : self.pad+1+self.pad, :]  # [bs,c,pad,w+2pad]
        lower_start = h - 2*self.pad - 1
        lower = padded[:, :, lower_start : lower_start+self.pad, :]     # [bs,c,pad,w+2pad]
        new_tensor = torch.cat([upper, padded, lower], dim=2)      # [bs,c,h+2pad,w+2pad]
        return new_tensor
    
    def pad_tensor_zero(self, tensor):
        return F.pad(tensor, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
    
    def transpose_qv(self, tensor):
        bs, c, h, w = tensor.shape
        return self.pad_tensor(tensor). \
                    unfold(2, self.n, 1).unfold(3, self.n, 1). \
                    permute(0, 2, 3, 1, 4, 5). \
                    reshape(bs, -1, c, self.n, self.n)

    def forward(self, x_rgb, x_ir):
        if self.spectrum == 'ir':
            bs, c, h, w = x_ir.shape
            ir_k = self.ir_k(x_ir) 
            ir_v = self.ir_v(x_ir)
            ir_q = self.ir_q(x_rgb - x_ir)
                        
            ir_mask = self.transpose_qv(ir_q) * self.transpose_k(ir_k)   # [bs, hw, c, n, n] * [bs, hw, c, 1, 1] -> [bs, hw, c, n, n]            
            ir_mask = ir_mask.view(bs, h*w, self.heads, self.head_dim, self.n, self.n).permute(0, 2, 1, 3, 4, 5) # [bs, heads, hw, head_dim, n, n]            
            
            # ir_mask = ir_mask.sum(3).view(bs, self.heads, h * w, 1, -1)  # 沿head_dim求和 [bs, heads, hw, 1, n*n]            
            # ir_mask = self.softmax(ir_mask).view(bs, self.heads, h * w, 1, self.n, self.n)  # [bs, heads, hw, 1, n, n]           
            ir_mask = ir_mask.view(bs, self.heads, h * w, self.head_dim, -1)  # 沿head_dim求和 [bs, heads, hw, 1, n*n]
            ir_mask = self.softmax(ir_mask).view(bs, self.heads, h * w, self.head_dim, self.n, self.n)  # [bs, heads, hw, 1, n, n]            
            
            ir_refine = self.transpose_qv(ir_v).view(bs, h*w, self.heads, self.head_dim, self.n, self.n).permute(0, 2, 1, 3, 4, 5) * \
                        ir_mask                # [bs, heads, hw, head_dim, n, n]            
            ir_refine = ir_refine.sum((-2, -1)) * self.gamma1            # [bs, heads, hw, head_dim]            
            out_refine = ir_refine.permute(0, 1, 3, 2).contiguous().view(bs, c, h, w) # [bs, heads, head_dim, hw]->[bs, c, h, w]
        elif self.spectrum == 'rgb':
            bs, c, h, w = x_rgb.shape
            rgb_k = self.rgb_k(x_rgb) 
            rgb_v = self.rgb_v(x_rgb)
            rgb_q = self.rgb_q(x_ir - x_rgb)

            rgb_mask = self.transpose_qv(rgb_q) * self.transpose_k(rgb_k)  # [bs, hw, c, n, n] * [bs, hw, c, 1, 1] -> [bs, hw, c, n, n]
            rgb_mask = rgb_mask.view(bs, -1, self.heads, self.head_dim, self.n, self.n).permute(0, 2, 1, 3, 4, 5) # [bs, heads, hw, head_dim, n, n]
            
            # rgb_mask = rgb_mask.sum(3).view(bs, self.heads, h * w, 1, -1)  # 沿head_dim求和 [bs, heads, hw, head_dim, n*n]
            # rgb_mask = self.softmax(rgb_mask).view(bs, self.heads, h * w, 1, self.n, self.n)  # [bs, heads, hw, 1, n, n]
            rgb_mask = rgb_mask.view(bs, self.heads, h * w, self.head_dim, -1)  # 沿head_dim求和 [bs, heads, hw, head_dim, n*n]
            rgb_mask = self.softmax(rgb_mask).view(bs, self.heads, h * w, self.head_dim, self.n, self.n)  # [bs, heads, hw, 1, n, n]
            
            rgb_refine = self.transpose_qv(rgb_v).view(bs, h*w, self.heads, self.head_dim, self.n, self.n).permute(0, 2, 1, 3, 4, 5) * \
                         rgb_mask               # [bs, heads, hw, head_dim, n, n]
            rgb_refine = rgb_refine.sum((-2, -1)) * self.gamma2            # [bs, heads, hw, head_dim]
            out_refine = rgb_refine.permute(0, 1, 3, 2).contiguous().view(bs, c, h, w) # [bs, heads, head_dim, hw]->[bs, c, h, w]
    
        return out_refine
    
class MyModuleBlock(nn.Module):
    def __init__(self, 
                 c_in,
                 n = 5,
                 spectrum = None):
        super().__init__()
        
        self.n = n
        self.block = MyMultiAttention(c_in, n = self.n, spectrum=spectrum)
        # self.block = CrissCrossAttentionBlock(c_in, 2, spectrum)
        # self.block = LePEAttentionBlock(c_in, spectrum)
        # self.block = WindowAttentionBlock(c_in, spectrum)
        # self.block = ABlock(c_in, spectrum)
        
        self.pool = nn.ModuleList([nn.AvgPool2d(2, 2), nn.MaxPool2d(2, 2)])
        
        self.pool_weight = LearnableWeights()
        
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.ones(1))
        self.conv = BasicConv2(c_in)
                        
    def forward(self, x_rgb, x_ir):
        b, c, h, w = x_ir.shape
        x_rgb = self.pool_weight(self.pool[0](x_rgb), self.pool[1](x_rgb))
        x_ir = self.pool_weight(self.pool[0](x_ir), self.pool[1](x_ir))
        
        x1 = self.block(x_rgb, x_ir)
        x2 = x1 + x_ir * self.gamma1
        x3 = self.conv(x2) * self.gamma3 + x2 * self.gamma2
        
        x_out =  F.interpolate(x3, size=([h, w]), mode='bilinear') 
        return x_out        

class MyModule(nn.Module):
    def __init__(self, 
                 c_in,
                 ):
        super().__init__()
        
        self.i2f_block = MyModuleBlock(c_in, n = 7, spectrum = 'ir')
        self.r2f_block = MyModuleBlock(c_in, n = 7, spectrum = 'rgb')
        
        self.output = BasicConv3(2 * c_in)
        
    def forward(self, x):
        x_rgb, x_ir = x
        
        # 注意力增强(互增强)
        x_ir_augmentation = self.i2f_block(x_rgb, x_ir)
        x_rgb_augmentation = self.r2f_block(x_ir, x_rgb)
        
        x_ir_out = x_ir_augmentation + x_ir        
        x_rgb_out = x_rgb_augmentation + x_rgb
        
        # 模态融合
        x_cat = torch.cat([torch.mul(x_rgb_out, x_ir_out), torch.max(x_rgb_out, x_ir_out)], dim = 1)
        x_out = self.output(x_cat)
        return x_out, x_rgb_out, x_ir_out
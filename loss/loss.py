class _Loss(object):
  def __init__(self, directory = 0):
    self.directory = directory

  def call(self, input, target):
    raise NotImplementedError

  def __call__(self, input, target):
    return self.call(input, target) if self.directory == 0 else self.call(target, input)

class KLDivLoss(_Loss):
  def __init__(self, directory = 0):
    super(KLDivLoss, self).__init__(directory)
    self.name = 'KLDiv'
  
  def call(self, input, target):
    return (target * (target / input).log()).sum(dim = 1).mean()



class CELoss(_Loss):
  def __init__(self, directory = 0):
    super(CELoss, self).__init__(directory)
    self.name = 'CE'

  def call(self, input, target):
    return -(target * input.log()).sum(dim = 1).mean()

class JSDivLoss(_Loss):
  def __init__(self, directory = 0):
    super(JSDivLoss, self).__init__(directory)
    self.kl_div = KLDivLoss()
    self.name = 'JSDiv'

  def call(self, input, target):
    p = (input + target) / 2
    return 1 / 2 * (self.kl_div(p, input) + self.kl_div(p, target))

class SymKLDivLoss(_Loss):
  def __init__(self, directory = 0):
    super(SymKLDivLoss, self).__init__(directory)
    self.kl_div = KLDivLoss()
    self.name = 'SymKLDiv'

  def call(self, input, target):
    return 1 / 2 * (self.kl_div(input, target) + self.kl_div(target, input))


loss_dict = {
  'kldiv': KLDivLoss,
  'ce': CELoss,
  'jsdiv': JSDivLoss,
  'symkldiv': SymKLDivLoss
}

def get_loss(name):
  name_dir = name.split(':')
  name = name_dir[0]
  directory = 0
  if len(name_dir) > 1:
    directory = int(name_dir[1])
  if name.lower() in loss_dict:
    return loss_dict[name.lower()](directory)
  else:
    raise ValueError('Undefined loss: %s'%(name))

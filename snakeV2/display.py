import pygame

class Render:
  def __init__(self, speed = 1, verbose = 0):
    self.verbose = verbose
    if self.verbose == 1:
      print('Render init')
    self.color_green = (0, 255, 0)
    self.color_red = (255, 0, 0)
    self.color_black = (0, 0, 0)
    self.color_white = (255, 255, 255)
    self.color_yellow = (255,220,50)
    self.color_orange = (255,165,0)
    self.display_width = 500
    self.display_height = 500
    self.cell_size = 10
    self.speed = speed
    pygame.init()
    self.display = pygame.display.set_mode((self.display_width, self.display_height))
    self.clock=pygame.time.Clock()

  def draw_step(self, observation, env):
    self.display.fill(self.color_white)
    pygame.display.set_caption("SCORE: " + str(env.score))

    for i in range(env.field_height):
      for j in range(env.field_width):
        if observation[j][i] > 0.0:
          # print('x:',j,'y:',i,'value:',obs[j][i])
          val = observation[j][i]
          color = self.color_white
          if val == 0.1:
            color = self.color_orange
          if val == 1.0:
            color = self.color_green
          if val == 0.2:
            color = self.color_yellow
          self.__draw(j,i,color)

    pygame.display.update()
    self.clock.tick(self.speed)

  def __draw(self, x, y, color):
    pygame.draw.rect(self.display, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
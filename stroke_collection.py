import pygame

pygame.init()

screen_size = (800,600)

screen = pygame.display.set_mode(screen_size)

still = True

white = (255,255,255)
grey = (180,180,180)
black = (0,0,0)

screen.fill(white)

clock = pygame.time.Clock()


mouse_down = False
mouse_prev = pygame.mouse.get_pos()
canvas_left, canvas_right, canvas_top, canvas_bottom = 100,700,250,500

white_canvas_left, white_canvas_right, white_canvas_top, white_canvas_bottom = 110,690,260,490

file = open("mouse_data.csv","w")
pygame.draw.rect(screen,black,pygame.Rect(canvas_left,canvas_top,canvas_right-canvas_left,canvas_bottom-canvas_top))
pygame.draw.rect(screen,white,pygame.Rect(white_canvas_left,white_canvas_top,white_canvas_right-white_canvas_left,white_canvas_bottom-white_canvas_top))
pygame.draw.rect(screen,grey,pygame.Rect(40,50,150,50))
pygame.draw.rect(screen,grey,pygame.Rect(40,150,150,50))
font = pygame.font.SysFont("Corbel",40)
text = font.render("Clear",False,black)
guess_text = font.render("Guess",False,black)
screen.blit(text, (60,55))
screen.blit(guess_text,(60,155))
def mouse_on_button(button_left, width, button_top, height):
    pos = pygame.mouse.get_pos() 
    if pos[0]>=button_left and pos[0]<=button_left+width and pos[1]>=button_top and pos[1]<=button_top+height:
        return True
    return False
while still:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            still = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
    
    if mouse_down:
        mouse_x, mouse_y = pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1] 
        cond1 = mouse_x>canvas_left and mouse_x<canvas_right and mouse_y>canvas_top and mouse_y < canvas_bottom
        cond2 = mouse_prev[0]>canvas_left and mouse_prev[0]<canvas_right and mouse_prev[1]>canvas_top and mouse_prev[1] < canvas_bottom
        if mouse_on_button(40,150,50,50):
            pygame.draw.rect(screen,white,pygame.Rect(white_canvas_left,white_canvas_top,white_canvas_right-white_canvas_left,white_canvas_bottom-white_canvas_top))
            file = open("mouse_data.csv","w")
        if mouse_on_button(40,150,150,50):
            file.close()
            file = open("mouse_data.csv","a")
        if cond1 and cond2:
            pygame.draw.line(screen,black,mouse_prev,pygame.mouse.get_pos(),2)
            normalized_pos = [(mouse_x-canvas_left)/(canvas_right-canvas_left),(mouse_y-canvas_top)/(canvas_bottom-canvas_top)]
            file.write(f"{[normalized_pos[0],normalized_pos[1],0]}\n")

    mouse_prev = pygame.mouse.get_pos()

    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
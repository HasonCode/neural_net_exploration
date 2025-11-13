import pygame

pygame.init()

screen_size = (1200,800)

screen = pygame.display.set_mode(screen_size)

still = True

white = (255,255,255)
black = (0,0,0)

screen.fill(white)

clock = pygame.time.Clock()

mouse_down = False
mouse_prev = pygame.mouse.get_pos()
file = open("mouse_data.csv","w")
while still:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            still = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
    
    if mouse_down: 
        pygame.draw.line(screen,black,mouse_prev,pygame.mouse.get_pos(),2)
        file.write(f"{[pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1],0]}\n")
    
    mouse_prev = pygame.mouse.get_pos()
    

    pygame.display.flip()
    clock.tick(400)

pygame.quit()
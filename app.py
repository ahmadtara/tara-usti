import pygame
import random
import sys

pygame.init()

# Ukuran layar
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Frog Game")

# Warna
GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Karakter Katak
frog_size = 40
frog_x = WIDTH // 2
frog_y = HEIGHT - frog_size
frog_speed = 40

# Mobil
car_width = 60
car_height = 40
car_speed = 5
cars = []

# Font
font = pygame.font.SysFont("Arial", 30)
score = 0

clock = pygame.time.Clock()

def spawn_car():
    lane_y = random.choice([200, 250, 300, 350])
    direction = random.choice(["left", "right"])
    if direction == "left":
        x = WIDTH
        speed = -car_speed
    else:
        x = -car_width
        speed = car_speed
    cars.append({"x": x, "y": lane_y, "speed": speed})

while True:
    screen.fill(BLUE)

    # Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and frog_x > 0:
        frog_x -= frog_speed
    if keys[pygame.K_RIGHT] and frog_x < WIDTH - frog_size:
        frog_x += frog_speed
    if keys[pygame.K_UP] and frog_y > 0:
        frog_y -= frog_speed
    if keys[pygame.K_DOWN] and frog_y < HEIGHT - frog_size:
        frog_y += frog_speed

    # Spawn mobil
    if random.randint(1, 60) == 1:
        spawn_car()

    # Update mobil
    for car in cars:
        car["x"] += car["speed"]

    # Hapus mobil di luar layar
    cars = [c for c in cars if -car_width < c["x"] < WIDTH + car_width]

    # Gambar mobil
    for car in cars:
        pygame.draw.rect(screen, RED, (car["x"], car["y"], car_width, car_height))
        # Cek tabrakan
        if (frog_x < car["x"] + car_width and
            frog_x + frog_size > car["x"] and
            frog_y < car["y"] + car_height and
            frog_y + frog_size > car["y"]):
            frog_x = WIDTH // 2
            frog_y = HEIGHT - frog_size
            score = 0  # Reset skor

    # Cek kalau katak sampai atas
    if frog_y <= 50:
        score += 1
        frog_x = WIDTH // 2
        frog_y = HEIGHT - frog_size

    # Gambar katak
    pygame.draw.rect(screen, GREEN, (frog_x, frog_y, frog_size, frog_size))

    # Gambar skor
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

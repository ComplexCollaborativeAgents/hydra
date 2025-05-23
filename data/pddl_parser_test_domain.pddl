﻿(define (domain angry_birds_scaled)
    (:requirements :typing :durative-actions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)
    (:types bird pig block platform)
    (:predicates (bird_released ?b - bird) (bird_dead ?b - bird) (pig_dead ?p - pig) (angle_adjusted))
    (:functions (x_bird ?b - bird) (y_bird ?b - bird) (v_bird ?b - bird) (vy_bird ?b - bird) (bird_id ?b - bird)
                (gravity) (angle_rate) (angle) (active_bird)
                (x_pig ?p - pig) (y_pig ?p - pig) (margin_pig ?p - pig)
                (x_platform ?pl - platform) (y_platform ?pl - platform) (platform_width ?pl - platform) (platform_height ?pl - platform)
                (x_block ?bl - block) (y_block ?bl - block) (block_width ?bl - block) (block_height ?bl - block) (block_life ?bl - block) (block_mass ?bl - block)
                ;; WOOD LIFE = 0.75   WOOD MASS COEFFICIENT = 0.375 ;; ICE LIFE = 0.75   ICE MASS COEFFICIENT = 0.375 ;; STONE LIFE = 1.2   STONE MASS COEFFICIENT = 0.375
                ;; WOOD LIFE MULTIPLIER = 1.0 ;; ICE LIFE MULTIPLIER = 0.5 ;; STONE LIFE MULTIPLIER = 2.0
                ;; THESE VALUES NEED TO BE VERIFIED
                ;; SEEMS LIKE THE SIZE AND SHAPE OF EACH BLOCK HAVE DIFFERENT LIFE VALUES WHICH ARE THEN MULTIPLIED BY THE MATERIAL LIFE MULTIPLIER
                ;; FOR NOW I WILL ASSIGN BLOCK_LIFE == 1.0 * MATERIAL_MULTIPLIER WHICH WILL BE ASSIGNED AUTOMATICALLY IN THE PROBLEM FILE VIA HYDRA.
                

    )

    (:process increasing_angle
        :parameters (?b - bird)
        :precondition (and
            (not (angle_adjusted))
            (= (active_bird) (bird_id ?b))
            (not (bird_dead ?b))
            (not (bird_released ?b))
            (<= (angle) 90)
        )
        :effect (and
            (increase (angle) (* #t (angle_rate)))
        )
    )

    (:process flying
        :parameters (?b - bird)
        :precondition (and
            (bird_released ?b)
            (= (active_bird) (bird_id ?b))
            (> (y_bird ?b) 0)
            (not (bird_dead ?b))
        )
        :effect (and
            ; (increase (y_bird ?b) (* #t (* 0.5 (vy_bird ?b))))
            (decrease (vy_bird ?b) (* #t (* 1.0 (gravity)) ))
            (increase (y_bird ?b) (* #t (* 1.0 (vy_bird ?b))))
            (increase (x_bird ?b) (* #t (* (v_bird ?b) (- 1 (/ (* (* (angle) 0.0174533) (* (angle) 0.0174533) ) 2) ) ) ))
        )
    )

    (:action pa-twang
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_dead ?b))
            (not (angle_adjusted))
            (not (bird_released ?b))

        )
        :effect (and
            (assign (vy_bird ?b) (* (v_bird ?b) (/ (* (* 4 (angle)) (- 180 (angle))) (- 40500 (* (angle) (- 180 (angle)))) )  ) )
            (bird_released ?b)
            (angle_adjusted)
        )
    )

    (:event collision_ground
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_dead ?b))
            (<= (y_bird ?b) 0)
        )
        :effect (and
            (bird_dead ?b)
            (assign (v_bird ?b) 0)
        )
    )

    (:event explode_bird
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (bird_dead ?b)
            (angle_adjusted)
        )
        :effect (and
            (assign (angle) 0)
            (assign (active_bird) (+ (active_bird) 1) )
            (not (angle_adjusted))
        )
    )

    (:event collision_pig
        :parameters (?b - bird ?p - pig)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (pig_dead ?p))
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_pig ?p) (margin_pig ?p)) )
            (>= (x_bird ?b) (x_pig ?p))
            (>= (y_bird ?b) (- (y_pig ?p) (margin_pig ?p)) )
            (<= (y_bird ?b) (y_pig ?p))
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (pig_dead ?p)
            (bird_dead ?b)
        )
    )

    (:event collision_block
        :parameters (?b - bird ?bl - block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (block_life ?bl) 0)
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_block ?bl) (block_width ?bl)) )
            (>= (x_bird ?b) (x_block ?bl))
            (>= (y_bird ?b) (- (y_block ?bl) (block_height ?bl)) )
            (<= (y_bird ?b) (y_block ?bl))
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (bird_dead ?b)
            (assign (block_life ?bl) (- (block_life ?bl) (v_bird ?b)) )
        )
    )

    (:event collision_platform
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_platform ?pl) (platform_width ?pl)) )
            (>= (x_bird ?b) (x_platform ?pl))
            (>= (y_bird ?b) (- (y_platform ?pl) (platform_height ?pl)) )
            (<= (y_bird ?b) (y_platform ?pl))
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (bird_dead ?b)
        )
    )


)


(define (domain angry_birds_scaled)
    (:requirements :typing :durative-actions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)
    (:types bird pig block platform)
    (:predicates (bird_released ?b - bird) (bird_in_slingshot) (pig_dead ?p - pig) (bird_dead ?b - bird) (angle_adjusted) (block_destroyed ?bl - block) )
    (:functions (x_platform ?bl - platform) (y_platform ?bl - platform) (platform_width ?bl - platform) (platform_height ?bl - platform) (x_block ?bl - block) (y_block ?bl - block) (block_width ?bl - block) (block_height ?bl - block) (gravity) (angle_rate) (x_bird ?b - bird) (y_bird ?b - bird) (v_bird ?b - bird) (vy_bird ?b - bird) (x_pig ?p - pig) (y_pig ?p - pig) (margin_pig ?p - pig) (angle) )
    
    (:process increasing_angle
        :parameters ()
        :precondition (and 
            (not (angle_adjusted)) 
            (bird_in_slingshot) 
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
            (> (y_bird ?b) 0)
    	    (not (bird_dead ?b))
        )
        :effect (and
    		; (increase (y_bird ?b) (* #t (vy_bird ?b)))
    		(decrease (vy_bird ?b) (* #t (gravity)) )
    		(increase (y_bird ?b) (* #t (vy_bird ?b)))
        	(increase (x_bird ?b) (* #t (* (v_bird ?b) (/ (* (* 4 (angle)) (- 180 (angle))) (- 40500 (* (angle) (- 180 (angle)))) ) ) ))
        )
    )
    
    (:action pa-twang
        :parameters (?b - bird)
        :precondition (and 
        	(bird_in_slingshot)
    		(not (angle_adjusted))
    		(not (bird_released ?b))
        )
        :effect (and 
    	    (assign (vy_bird ?b) (* (v_bird ?b) (- 1 (/ (* (* (angle) 0.0174533) (* (angle) 0.0174533) ) 2) )  ) )
            (bird_released ?b)
    	    (angle_adjusted)
    	    (not (bird_in_slingshot))
        )
    )
    
    (:event collision_pig
        :parameters (?b - bird ?p - pig)
        :precondition (and 
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_pig ?p) (margin_pig ?p)) )
            (>= (x_bird ?b) (x_pig ?p))
            (>= (y_bird ?b) (- (y_pig ?p) (margin_pig ?p)) )
            (<= (y_bird ?b) (y_pig ?p))
        )
        :effect (and 
            (assign (v_bird ?b) 0)
            (pig_dead ?p)
            (not (bird_released ?b))
        )
    )
    
    (:event collision_block
        :parameters (?b - bird ?bl - block)
        :precondition (and 
            (not (block_destroyed ?bl))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_block ?bl) (block_width ?bl)) )
            (>= (x_bird ?b) (x_block ?bl))
            (>= (y_bird ?b) (- (y_block ?bl) (block_height ?bl)) )
            (<= (y_bird ?b) (y_block ?bl))
        )
        :effect (and 
            (assign (v_bird ?b) 0)
            (block_destroyed ?bl)
            (not (bird_released ?b))
        )
    )

    (:event collision_platform
        :parameters (?b - bird ?pl - platform)
        :precondition (and
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
    
    (:event collision_ground
        :parameters (?b - bird)
        :precondition (and 
            (<= (y_bird ?b) 0)
        )
        :effect (and 
            (bird_dead ?b)
            (assign (v_bird ?b) 0)
        )
    )

)


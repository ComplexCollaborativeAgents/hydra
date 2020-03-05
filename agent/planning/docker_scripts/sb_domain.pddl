(define (domain angry_birds_scaled)
    (:requirements :typing :durative-actions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)
    (:types bird pig wood_block stone_block ice_block platform)
    (:predicates (bird_released ?b - bird) (bird_dead ?b - bird) (pig_dead ?p - pig) (angle_adjusted)
                 (wood_destroyed ?wbl - wood_block) (stone_destroyed ?sbl - stone_block) (ice_destroyed ?ibl - ice_block))
    (:functions (x_bird ?b - bird) (y_bird ?b - bird) (v_bird ?b - bird) (vy_bird ?b - bird) (bird_id ?b - bird)
                (gravity) (angle_rate) (angle) (active_bird)
                (x_pig ?p - pig) (y_pig ?p - pig) (margin_pig ?p - pig)
                (x_platform ?bl - platform) (y_platform ?bl - platform) (platform_width ?bl - platform) (platform_height ?bl - platform)
                (x_wood ?wbl - wood_block) (y_wood ?wbl - wood_block) (wood_width ?wbl - wood_block) (wood_height ?wbl - wood_block)
                (x_stone ?sbl - stone_block) (y_stone ?sbl - stone_block) (stone_width ?sbl - stone_block) (stone_height ?sbl - stone_block)
                (x_ice ?ibl - ice_block) (y_ice ?ibl - ice_block) (ice_width ?ibl - ice_block) (ice_height ?ibl - ice_block)
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

    (:event collision_wood
        :parameters (?b - bird ?wbl - wood_block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (wood_destroyed ?wbl))
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_wood ?wbl) (wood_width ?wbl)) )
            (>= (x_bird ?b) (x_wood ?wbl))
            (>= (y_bird ?b) (- (y_wood ?wbl) (wood_height ?wbl)) )
            (<= (y_bird ?b) (y_wood ?wbl))
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (bird_dead ?b)
            (wood_destroyed ?wbl)
        )
    )

    (:event collision_stone
        :parameters (?b - bird ?sbl - stone_block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (stone_destroyed ?sbl))
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_stone ?sbl) (stone_width ?sbl)) )
            (>= (x_bird ?b) (x_stone ?sbl))
            (>= (y_bird ?b) (- (y_stone ?sbl) (stone_height ?sbl)) )
            (<= (y_bird ?b) (y_stone ?sbl))
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (bird_dead ?b)
        )
    )

    (:event collision_ice
        :parameters (?b - bird ?ibl - ice_block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (ice_destroyed ?ibl))
            (not (bird_dead ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_ice ?ibl) (ice_width ?ibl)) )
            (>= (x_bird ?b) (x_ice ?ibl))
            (>= (y_bird ?b) (- (y_ice ?ibl) (ice_height ?ibl)) )
            (<= (y_bird ?b) (y_ice ?ibl))
        )
        :effect (and
            (assign (v_bird ?b) (* 0.5 (v_bird ?b)) )
            (assign (vy_bird ?b) (* 0.5 (vy_bird ?b)) )
            ; (bird_dead ?b)
            (ice_destroyed ?ibl)
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


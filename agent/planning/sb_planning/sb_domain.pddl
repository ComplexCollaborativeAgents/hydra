(define (domain angry_birds_scaled)
    (:requirements :typing :disjunctive-preconditions :fluents :time :negative-preconditions)
    (:types bird pig block platform)
    (:predicates (bird_released ?b - bird) (pig_dead ?p - pig) (angle_adjusted) (block_explosive ?bl - block) (pig_killed))
    (:functions (x_bird ?b - bird) (y_bird ?b - bird) (v_bird ?b - bird) (vx_bird ?b - bird) (vy_bird ?b - bird) (m_bird ?b - bird) (bird_id ?b - bird) (bounce_count ?b - bird)
                (bird_type ?b - bird) ;; BIRD TYPES: RED=0, YELLOW=1, BLACK=2, WHITE=3, BLUE=4 ;;
                (gravity) (angle_rate) (angle) (active_bird) (ground_damper) (max_angle) (gravity_factor)
                (base_life_wood_multiplier) (base_life_ice_multiplier) (base_life_stone_multiplier) (base_life_tnt_multiplier)
                (base_mass_wood_multiplier) (base_mass_ice_multiplier) (base_mass_stone_multiplier) (base_mass_tnt_multiplier)
                (meta_wood_multiplier) (meta_stone_multiplier) (meta_ice_multiplier)
                (v_bird_multiplier)
                (x_pig ?p - pig) (y_pig ?p - pig) (pig_radius ?p - pig) (m_pig ?p - pig)
                (x_platform ?pl - platform) (y_platform ?pl - platform) (platform_width ?pl - platform) (platform_height ?pl - platform)
                (x_block ?bl - block) (y_block ?bl - block) (block_width ?bl - block) (block_height ?bl - block) (block_life ?bl - block) (block_mass ?bl - block) (block_stability ?bl - block)
                (points_score)
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
            (not (bird_released ?b))
            (< (angle) (max_angle))
            (>= (angle) 0)
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
        )
        :effect (and
            (decrease (vy_bird ?b) (* #t (* 1.0 (gravity)) ))
            (increase (y_bird ?b) (* #t (* 1.0 (vy_bird ?b))))
            (increase (x_bird ?b) (* #t (* 1.0 (vx_bird ?b))))
        )
    )

    (:action pa-twang
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (angle_adjusted))
            (not (bird_released ?b))

        )
        :effect (and
            (assign (vy_bird ?b) (* (v_bird ?b) (/ (* (* 4 (angle)) (- 180 (angle))) (- 40500 (* (angle) (- 180 (angle)))) )  ) )
            (assign (vx_bird ?b) (* (v_bird ?b) (- 1 (/ (* (* (angle) 0.0174533) (* (angle) 0.0174533) ) 2) ) ) )
            (bird_released ?b)
            (angle_adjusted)
            ;(decrease (points_score) 10000)
        )
    )

    (:event collision_ground
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (<= (y_bird ?b) 0)
            (> (v_bird ?b) 20)
        )
        :effect (and
            (assign (y_bird ?b) 1)
            ; (assign (v_bird ?b) (* (v_bird ?b) 1.0) )
            ; (assign (vy_bird ?b) (* (v_bird ?b) (/ (* (* 4 (angle)) (- 180 (angle))) (- 40500 (* (angle) (- 180 (angle)))) )  ) )
            (assign (vy_bird ?b) (* (* (vy_bird ?b) -1) (ground_damper)))
            (assign (bounce_count ?b) (+ (bounce_count ?b) 1))

        )
    )

    (:event explode_bird
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (or (>= (bounce_count ?b) 3)
                (>= (x_bird ?b) 800)
            )
            (angle_adjusted)
            ; (<= (v_bird) 20)
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
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_pig ?p) (- (pig_radius ?p) (* (pig_radius ?p) 0.2)) ) )
            (>= (x_bird ?b) (- (x_pig ?p) (- (pig_radius ?p) (* (pig_radius ?p) 0.2)) ) )
            (>= (y_bird ?b) (- (y_pig ?p) (- (pig_radius ?p) (* (pig_radius ?p) 0.2)) ) )
            (<= (y_bird ?b) (+ (y_pig ?p) (- (pig_radius ?p) (* (pig_radius ?p) 0.2)) ) )
        )
        :effect (and
            (pig_dead ?p)
            (assign (vx_bird ?b) (- (vx_bird ?b) (* (/ (* 2 (m_pig ?p)) (+ (m_bird ?b) (m_pig ?p))) (*
                (/
                    (+ (+ (- (- (- (- (+ (* (vx_bird ?b) (x_bird ?b)) (* (vy_bird ?b) (y_bird ?b))) (* (vx_bird ?b) (x_pig ?p))) (* (vy_bird ?b) (y_pig ?p))) (* 0 (x_bird ?b))) (* 0 (y_bird ?b))) (* 0 (x_pig ?p))) (* 0 (y_pig ?p)))
                    (+ (+ (- (- (- (- (+ (* (x_pig ?p) (x_pig ?p)) (* (y_pig ?p) (y_pig ?p))) (* (x_pig ?p) (x_bird ?b))) (* (y_pig ?p) (y_bird ?b))) (* (x_bird ?b) (x_pig ?p))) (* (y_bird ?b) (y_pig ?p))) (* (x_bird ?b) (x_bird ?b))) (* (y_bird ?b) (y_bird ?b)))
                )
                (- (x_pig ?p) (x_bird ?b)) )
            )  ) )

            (assign (vy_bird ?b) (- (vy_bird ?b) (* (/ (* 2 (m_pig ?p)) (+ (m_bird ?b) (m_pig ?p))) (*
                (/
                    (+ (+ (- (- (- (- (+ (* (vx_bird ?b) (x_bird ?b)) (* (vy_bird ?b) (y_bird ?b))) (* (vx_bird ?b) (x_pig ?p))) (* (vy_bird ?b) (y_pig ?p))) (* 0 (x_bird ?b))) (* 0 (y_bird ?b))) (* 0 (x_pig ?p))) (* 0 (y_pig ?p)))
                    (+ (+ (- (- (- (- (+ (* (x_pig ?p) (x_pig ?p)) (* (y_pig ?p) (y_pig ?p))) (* (x_pig ?p) (x_bird ?b))) (* (y_pig ?p) (y_bird ?b))) (* (x_bird ?b) (x_pig ?p))) (* (y_bird ?b) (y_pig ?p))) (* (x_bird ?b) (x_bird ?b))) (* (y_bird ?b) (y_bird ?b)))
                )
                (- (y_pig ?p) (y_bird ?b)) )
            )  ) )

            (assign (bounce_count ?b) (+ (bounce_count ?b) 1))
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    (:event collision_block_bounce_off
        :parameters (?b - bird ?bl - block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (block_life ?bl) 0)
            (> (v_bird ?b) 0)
            (> (block_stability ?bl) (v_bird ?b) )
            (> (block_life ?bl) (v_bird ?b) )
            (<= (x_bird ?b) (+ (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (x_bird ?b) (- (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (y_bird ?b) (- (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (<= (y_bird ?b) (+ (y_block ?bl) (/ (block_height ?bl) 2) ) )
        )
        :effect (and
            (assign (block_stability ?bl) (- (block_stability ?bl) (v_bird ?b)) )
            (assign (block_life ?bl) (- (block_life ?bl) (v_bird ?b)) )
            (assign (x_bird ?b) (- (x_block ?bl) (+ (/ (block_width ?bl) 2) 1) ) )
            (assign (y_bird ?b) (+ (y_block ?bl) (+ (/ (block_height ?bl) 2) 1) ) )
            (assign (vy_bird ?b) (* (vy_bird ?b) (/ (v_bird ?b) (block_stability ?bl)) ))
            (assign (vx_bird ?b) (- 0 (* (vx_bird ?b) (/ (v_bird ?b) (block_stability ?bl))) ))
            (assign (v_bird ?b) (* (v_bird ?b) (/ (v_bird ?b) (block_stability ?bl)) ))
            (assign (bounce_count ?b) (+ (bounce_count ?b) 1))
        )
    )


    (:event collision_block
        :parameters (?b - bird ?bl - block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (block_life ?bl) 0)
            (> (v_bird ?b) 0)
            (or
            	(<= (block_stability ?bl) (v_bird ?b))
            	(<= (block_life ?bl) (v_bird ?b))
        	)
            (<= (x_bird ?b) (+ (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (x_bird ?b) (- (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (y_bird ?b) (- (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (<= (y_bird ?b) (+ (y_block ?bl) (/ (block_height ?bl) 2) ) )
        )
        :effect (and
            (assign (block_stability ?bl) (- (block_stability ?bl) (v_bird ?b)) )
            (assign (block_life ?bl) (- (block_life ?bl) (v_bird ?b)) )
            (assign (vy_bird ?b) (* (vy_bird ?b) 0.5))
            (assign (vx_bird ?b) (* (vx_bird ?b) 0.5))
            (assign (v_bird ?b) (* (v_bird ?b) 0.5))
            (assign (bounce_count ?b) (+ (bounce_count ?b) 1))
            ;(increase (points_score) 500)
        )
    )


    (:event remove_unsupported_block
        :parameters (?bl_bottom - block ?bl_top - block)
        :precondition (and
            (or  (<= (block_life ?bl_bottom) 0)
            (<= (block_stability ?bl_bottom) 0))
            (> (block_life ?bl_top) 0)
            (> (block_stability ?bl_top) 0)
            (<= (x_block ?bl_bottom) (+ (x_block ?bl_top) (/ (block_width ?bl_top) 2) ) )
            (>= (x_block ?bl_bottom) (- (x_block ?bl_top) (/ (block_width ?bl_top) 2) ) )
            (<= (y_block ?bl_bottom) (- (y_block ?bl_top) (/ (block_height ?bl_top) 2) ) )
        )
        :effect (and
            (assign (block_life ?bl_top) (- (block_life ?bl_top) 100) )
            (assign (y_block ?bl_top) (/ (block_height ?bl_top) 2) )
            (assign (block_stability ?bl_top) 0)
            (increase (points_score) 500)
        )
    )

    (:event explode_block
        :parameters (?bl_tnt - block ?bl_near - block)
        :precondition (and
            (block_explosive ?bl_tnt)
            ; (<= (block_stability ?bl_tnt) 0)
            (<= (block_life ?bl_tnt) 0)
            (> (block_stability ?bl_near) 0)
            (> (block_life ?bl_near) 0)
            (<= (- (x_block ?bl_tnt) (x_block ?bl_near)) 100 )
            (>= (- (x_block ?bl_tnt) (x_block ?bl_near)) -100 )
            (<= (- (y_block ?bl_tnt) (y_block ?bl_near)) 100 )
            (>= (- (y_block ?bl_tnt) (y_block ?bl_near)) -100 )
        )
        :effect (and
            (assign (block_life ?bl_near) 0)
            (assign (block_stability ?bl_near) 0)
            (increase (points_score) 1000)
        )
    )

    (:event explode_pig
        :parameters (?bl_tnt - block ?p - pig)
        :precondition (and
            (block_explosive ?bl_tnt)
            ; (<= (block_stability ?bl_tnt) 0)
            (<= (block_life ?bl_tnt) 0)
            (not (pig_dead ?p))
            (<= (- (x_block ?bl_tnt) (x_pig ?p)) 50 )
            (>= (- (x_block ?bl_tnt) (x_pig ?p)) -50 )
            (<= (- (y_block ?bl_tnt) (y_pig ?p)) 50 )
            (>= (- (y_block ?bl_tnt) (y_pig ?p)) -50 )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    (:event remove_unsupported_pig
        :parameters (?bl_bottom - block ?p - pig)
        :precondition (and
        	(not (pig_dead ?p))
            (or (< (block_life ?bl_bottom) 0)
            (<= (block_stability ?bl_bottom) 0))
            (<= (x_pig ?p) (+ (x_block ?bl_bottom) (/ (block_width ?bl_bottom) 2) ) )
            (>= (x_pig ?p) (- (x_block ?bl_bottom) (/ (block_width ?bl_bottom) 2) ) )
            (>= (y_pig ?p) (+ (y_block ?bl_bottom) (/ (block_height ?bl_bottom) 2) ) )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    (:event collision_platform
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (v_bird ?b) 0)
            (<= (x_bird ?b) (+ (x_platform ?pl) (/ (platform_width ?pl) 1.75) ) )
            (>= (x_bird ?b) (- (x_platform ?pl) (/ (platform_width ?pl) 1.75) ) )
            (>= (y_bird ?b) (- (y_platform ?pl) (/ (platform_height ?pl) 1.75) ) )
            (<= (y_bird ?b) (+ (y_platform ?pl) (/ (platform_height ?pl) 1.75) ) )
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (assign (vx_bird ?b) 0)
            (assign (bounce_count ?b) 3)
        )
    )

    (:event explode_block_from_bird
        :parameters (?b - bird ?bl_near - block)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
        	; (or
      			(= (bird_type ?b) 2) (> (bounce_count ?b) 0)
      			; (and (= (bird_type ?b) 3) (= (bounce_count ?b) 1) (bird_tapped ?b) )
  			; )
            (> (block_stability ?bl_near) 0)
            (> (block_life ?bl_near) 0)
            (<= (- (x_bird ?b) (x_block ?bl_near)) 70 )
            (>= (- (x_bird ?b) (x_block ?bl_near)) -70 )
            (<= (- (y_bird ?b) (y_block ?bl_near)) 70 )
            (>= (- (y_bird ?b) (y_block ?bl_near)) -70 )
        )
        :effect (and
            (assign (block_life ?bl_near) 0)
            (assign (block_stability ?bl_near) 0)
        )
    )

    (:event explode_pig_from_bird
        :parameters (?b - bird ?p - pig)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
        	; (or
      			(= (bird_type ?b) 2) (> (bounce_count ?b) 0)
      			; (and (= (bird_type ?b) 3) (= (bounce_count ?b) 1) (bird_tapped ?b) )
  			; )
            (not (pig_dead ?p))
            (<= (- (x_bird ?b) (x_pig ?p)) 70 )
            (>= (- (x_bird ?b) (x_pig ?p)) -70 )
            (<= (- (y_bird ?b) (y_pig ?p)) 70 )
            (>= (- (y_bird ?b) (y_pig ?p)) -70 )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

)

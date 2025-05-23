(define(domain angry_birds_scaled)
	(:requirements :typing :durative-actions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)
	(:types bird pig block platform)
	(:predicates
		(bird_released ?b - bird)
		(pig_dead ?p - pig)
		(angle_adjusted)
	)
	(:functions
		(x_bird ?b - bird)
		(y_bird ?b - bird)
		(v_bird ?b - bird)
		(vx_bird ?b - bird)
		(vy_bird ?b - bird)
		(m_bird ?b - bird)
		(bird_id ?b - bird)
		(bounce_count ?b - bird)
		(gravity)
		(angle_rate)
		(angle)
		(active_bird)
		(ground_damper)
		(x_pig ?p - pig)
		(y_pig ?p - pig)
		(pig_radius ?p - pig)
		(m_pig ?p - pig)
		(x_platform ?pl - platform)
		(y_platform ?pl - platform)
		(platform_width ?pl - platform)
		(platform_height ?pl - platform)
		(x_block ?bl - block)
		(y_block ?bl - block)
		(block_width ?bl - block)
		(block_height ?bl - block)
		(block_life ?bl - block)
		(block_mass ?bl - block)
	)
	(:process increasing_angle
		:parameters (?b - bird)
		:precondition (and 
		 (not (angle_adjusted) )
		 (= (active_bird)  (bird_id ?b) )
		 (not (bird_released ?b) )
		 (<= (angle)  90)
		) 
		:effect (and 
		 (increase (angle)  (* #t (angle_rate) )
)
		) 
	)  
	(:process flying
		:parameters (?b - bird)
		:precondition (and 
		 (bird_released ?b)
		 (= (active_bird)  (bird_id ?b) )
		 (> (y_bird ?b)  0)
		) 
		:effect (and 
		 (decrease (vy_bird ?b)  (* #t (* 1.0 (gravity) )
)
)
		 (increase (y_bird ?b)  (* #t (* 1.0 (vy_bird ?b) )
)
)
		 (increase (x_bird ?b)  (* #t (* 1.0 (vx_bird ?b) )
)
)
		) 
	)  
	(:event collision_ground
		:parameters (?b - bird)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (<= (y_bird ?b)  0)
		 (> (v_bird ?b)  20)
		) 
		:effect (and 
		 (assign (y_bird ?b)  1)
		 (assign (vy_bird ?b)  (* (* (vy_bird ?b)  -1)
 (ground_damper) )
)
		 (assign (bounce_count ?b)  (+ (bounce_count ?b)  1)
)
		) 
	)  
	(:event explode_bird
		:parameters (?b - bird)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (bird_released ?b)
		 (>= (bounce_count ?b)  3)
		 (angle_adjusted)
		) 
		:effect (and 
		 (assign (angle)  0)
		 (assign (active_bird)  (+ (active_bird)  1)
)
		 (not (angle_adjusted) )
		) 
	)  
	(:event collision_pig
		:parameters (?b - bird ?p - pig)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (not (pig_dead ?p) )
		 (> (v_bird ?b)  0)
		 (<= (x_bird ?b)  (+ (x_pig ?p)  (- (pig_radius ?p)  (* (pig_radius ?p)  0.2)
)
)
)
		 (>= (x_bird ?b)  (- (x_pig ?p)  (- (pig_radius ?p)  (* (pig_radius ?p)  0.2)
)
)
)
		 (>= (y_bird ?b)  (- (y_pig ?p)  (- (pig_radius ?p)  (* (pig_radius ?p)  0.2)
)
)
)
		 (<= (y_bird ?b)  (+ (y_pig ?p)  (- (pig_radius ?p)  (* (pig_radius ?p)  0.2)
)
)
)
		) 
		:effect (and 
		 (pig_dead ?p)
		 (assign (vx_bird ?b)  (- (vx_bird ?b)  (* (/ (* 2 (m_pig ?p) )
 (+ (m_bird ?b)  (m_pig ?p) )
)
 (* (/ (+ (+ (- (- (- (- (+ (* (vx_bird ?b)  (x_bird ?b) )
 (* (vy_bird ?b)  (y_bird ?b) )
)
 (* (vx_bird ?b)  (x_pig ?p) )
)
 (* (vy_bird ?b)  (y_pig ?p) )
)
 (* 0 (x_bird ?b) )
)
 (* 0 (y_bird ?b) )
)
 (* 0 (x_pig ?p) )
)
 (* 0 (y_pig ?p) )
)
 (+ (+ (- (- (- (- (+ (* (x_pig ?p)  (x_pig ?p) )
 (* (y_pig ?p)  (y_pig ?p) )
)
 (* (x_pig ?p)  (x_bird ?b) )
)
 (* (y_pig ?p)  (y_bird ?b) )
)
 (* (x_bird ?b)  (x_pig ?p) )
)
 (* (y_bird ?b)  (y_pig ?p) )
)
 (* (x_bird ?b)  (x_bird ?b) )
)
 (* (y_bird ?b)  (y_bird ?b) )
)
)
 (- (x_pig ?p)  (x_bird ?b) )
)
)
)
)
		 (assign (vy_bird ?b)  (- (vy_bird ?b)  (* (/ (* 2 (m_pig ?p) )
 (+ (m_bird ?b)  (m_pig ?p) )
)
 (* (/ (+ (+ (- (- (- (- (+ (* (vx_bird ?b)  (x_bird ?b) )
 (* (vy_bird ?b)  (y_bird ?b) )
)
 (* (vx_bird ?b)  (x_pig ?p) )
)
 (* (vy_bird ?b)  (y_pig ?p) )
)
 (* 0 (x_bird ?b) )
)
 (* 0 (y_bird ?b) )
)
 (* 0 (x_pig ?p) )
)
 (* 0 (y_pig ?p) )
)
 (+ (+ (- (- (- (- (+ (* (x_pig ?p)  (x_pig ?p) )
 (* (y_pig ?p)  (y_pig ?p) )
)
 (* (x_pig ?p)  (x_bird ?b) )
)
 (* (y_pig ?p)  (y_bird ?b) )
)
 (* (x_bird ?b)  (x_pig ?p) )
)
 (* (y_bird ?b)  (y_pig ?p) )
)
 (* (x_bird ?b)  (x_bird ?b) )
)
 (* (y_bird ?b)  (y_bird ?b) )
)
)
 (- (y_pig ?p)  (y_bird ?b) )
)
)
)
)
		 (assign (bounce_count ?b)  (+ (bounce_count ?b)  1)
)
		) 
	)  
	(:event collision_block
		:parameters (?b - bird ?bl - block)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (> (block_life ?bl)  0)
		 (> (v_bird ?b)  0)
		 (<= (x_bird ?b)  (+ (x_block ?bl)  (/ (block_width ?bl)  2)
)
)
		 (>= (x_bird ?b)  (- (x_block ?bl)  (/ (block_width ?bl)  2)
)
)
		 (>= (y_bird ?b)  (- (y_block ?bl)  (/ (block_height ?bl)  2)
)
)
		 (<= (y_bird ?b)  (+ (y_block ?bl)  (/ (block_height ?bl)  2)
)
)
		) 
		:effect (and 
		 (assign (v_bird ?b)  0)
		 (assign (block_life ?bl)  (- (block_life ?bl)  (v_bird ?b) )
)
		 (assign (bounce_count ?b)  3)
		) 
	)  
	(:event collision_platform
		:parameters (?b - bird ?pl - platform)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (> (v_bird ?b)  0)
		 (<= (x_bird ?b)  (+ (x_platform ?pl)  (/ (platform_width ?pl)  2)
)
)
		 (>= (x_bird ?b)  (- (x_platform ?pl)  (/ (platform_width ?pl)  2)
)
)
		 (>= (y_bird ?b)  (- (y_platform ?pl)  (/ (platform_height ?pl)  2)
)
)
		 (<= (y_bird ?b)  (+ (y_platform ?pl)  (/ (platform_height ?pl)  2)
)
)
		) 
		:effect (and 
		 (assign (v_bird ?b)  0)
		 (assign (bounce_count ?b)  3)
		) 
	)  
	(:action pa-twang
		:parameters (?b - bird)
		:precondition (and 
		 (= (active_bird)  (bird_id ?b) )
		 (not (angle_adjusted) )
		 (not (bird_released ?b) )
		) 
		:effect (and 
		 (assign (vy_bird ?b)  (* (v_bird ?b)  (/ (* (* 4 (angle) )
 (- 180 (angle) )
)
 (- 40500 (* (angle)  (- 180 (angle) )
)
)
)
)
)
		 (assign (vx_bird ?b)  (* (v_bird ?b)  (- 1 (/ (* (* (angle)  0.0174533)
 (* (angle)  0.0174533)
)
 2)
)
)
)
		 (bird_released ?b)
		 (angle_adjusted)
		) 
	)  
)

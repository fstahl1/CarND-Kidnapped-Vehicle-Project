/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;

  for (int i = 0; i < num_particles; ++i) {
    
    particles[i].x += (velocity/yaw_rate) * 
                        ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta) );
    std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    particles[i].x = dist_x(gen);

    particles[i].y += (velocity/yaw_rate) * 
                        ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t) );
    std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    particles[i].y = dist_y(gen);

    particles[i].theta += yaw_rate*delta_t;
    std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);  
    particles[i].theta = dist_theta(gen);
  }

}

LandmarkObs transform(double x_p, double y_p, double x_c, double y_c, double theta) {
  LandmarkObs transformed;
  transformed.x = cos(theta)*x_c - sin(theta)*y_c + x_p;
  transformed.y = sin(theta)*x_c + cos(theta)*y_c + y_p;
  return transformed;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (std::size_t o = 0; o < observations.size(); ++o) {
    for (std::size_t p = 0; p < predicted.size(); ++p) {
      // initialize minimum distance with large value
      double min_dist = 100000;
      double distance = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);
      // if (dist <= sensor_range) {
        // check if current distance is smaller than all previous distances
        if (distance < min_dist) {
          min_dist = distance;
          // set id from observation to map landmark id
          observations[o].id = predicted[p].id;          
        }
      // }
    }
  }

}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // compute predicted observations from map landmarks
  vector<LandmarkObs> predictions;
  int temp_id;
  double temp_x;
  double temp_y;
  for (unsigned int m = 0; m < map_landmarks.landmark_list.size(); ++m) {
    temp_id = map_landmarks.landmark_list[m].id_i;
    temp_x = map_landmarks.landmark_list[m].x_f;
    temp_y = map_landmarks.landmark_list[m].y_f;
    predictions.push_back(LandmarkObs{temp_id, temp_x, temp_y});
  }

  // iterate over all particles
  for (int p = 0; p < num_particles; ++p) {
    // initialize vector for transformed observations of current particle
    vector<LandmarkObs> transformed_observations;

    LandmarkObs temp_obs;
    // iterate over all observations and transform them into map coordinates
    for (unsigned int o = 0; o < observations.size(); ++o) {
      temp_obs = transform(particles[p].x, particles[p].y, observations[o].x, observations[o].y, particles[p].theta);
      // temp_obs.id = observations[o].id; // does not have an id yet
      transformed_observations.push_back(temp_obs);
    }

    // associate all observations with predicted observations (landmark positions) for current particle
    dataAssociation(predictions, transformed_observations);

        // iterate over all observations
    for (unsigned int o = 0; o < observations.size(); ++o) {

      double curr_pred_x, curr_pred_y;
      for (unsigned int m = 0; m < predictions.size(); ++m) {
        if (predictions[m].id == observations[o].id) {
          curr_pred_x = predictions[m].x;
          curr_pred_y = predictions[m].y;
          break;
        }
      }

      double gauss = multiv_prob(std_landmark[0], std_landmark[1], observations[o].x, observations[o].y, curr_pred_x, curr_pred_y);

      particles[p].weight *= gauss;

    }
    
  }

}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
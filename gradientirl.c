#include "math.h"
#include "string.h"

long double pnorm_max(int n_actions, double approx_level, double*value_set)
{
  long double sum_value=0;
  int i=0;
  for(i=0;i<n_actions;i++)
  {
    sum_value=sum_value+powl(value_set[i],approx_level);
  }
  return powl(sum_value,1.0/approx_level);
}
void pnorm_V_iteration(int n_states, int n_actions, double* transition_probabilities, double* reward, double discount, double threshold, double approx_level, double*V, double* new_V, double*value_update, double*Q)
{
  // Value iteration
  double diff=1;
  while(diff>threshold)
  {
    int i,j,k;
    for(i=0;i<n_states;i++)
    {
      for(j=0; j<n_actions;j++)
      {
        value_update[j]=0;
        for(k=0;k<n_states;k++)
        {
          value_update[j]+=transition_probabilities[i*n_states*n_actions+j*n_states+k]*(reward[k]+discount*V[k]);
        }
      }
      new_V[i]=pnorm_max(n_actions,approx_level,value_update);
    }
    diff=0;
    for(i=0;i<n_states;i++)
    {
      if(fabs(new_V[i]-V[i])>threshold)
      {
        diff=1;
        break;
      }
    }
    if(diff==0)
    {
      break;
    }
   memcpy(V,new_V,n_states*sizeof(double));
  }


  // Q value iteration
  int i,j,k;
  for(i=0;i<n_states;i++)
    for(j=0;j<n_actions;j++)
    {
      Q[i*n_actions+j]=0;
      for(k=0;k<n_states;k++)
        Q[i*n_actions+j]=Q[i*n_actions+j]+transition_probabilities[i*n_states*n_actions+j*n_states+k]*(reward[k]+discount*V[k]);
    }

}
void pnorm_G_iteration(int n_states, int n_actions, int n_features, double*featureMatrix, double* transition_probabilities, double* reward, double discount, double threshold, double approx_level, double*Values,double*QValues, double*V_Gradients, double*Q_Gradients,double*d_pnorm_vector,double*d_summation_matrix,double*new_V_Gradients,double*d_sum_q)
{

  //V Gradient iteration
  double diff=1;
  // Precompute several important variables
  int i,j,k;
  for(i=0;i<n_states;i++)
  {
    // d_pnorm_vector
    long double sum_value=0;
    for(j=0;j<n_actions;j++)
    {
      sum_value=sum_value+powl(QValues[i*n_actions+j],approx_level);
    }
    long double d_pnorm=powl(sum_value,1.0/approx_level-1);
    // d_summation_matrix
    for(j=0;j<n_actions;j++)
    {
     d_summation_matrix[i*n_actions+j]=d_pnorm*powl(QValues[i*n_actions+j],approx_level-1);
    }
  }
  while(diff>threshold)
  {
    int i,j,k,f;
    for(i=0;i<n_states;i++)
    {
      for(f=0;f<n_features;f++)
        d_sum_q[f]=0;
      for(j=0; j<n_actions;j++)
      {
        double d_summation=d_summation_matrix[i*n_actions+j];
        for(k=0;k<n_states;k++)
        {
          // feature summation
          for(f=0;f<n_features;f++)
            d_sum_q[f]=d_sum_q[f]+d_summation*transition_probabilities[i*n_states*n_actions+j*n_states+k]*(featureMatrix[k*n_features+f]+discount*V_Gradients[k*n_features+f]);
        }
      }
      for(f=0;f<n_features;f++)
        new_V_Gradients[i*n_features+f]=d_sum_q[f];
    }
    diff=0;
    for(i=0;i<n_states;i++)
    {
      for(f=0;f<n_features;f++)
        if(fabs(new_V_Gradients[i*n_features+f]-V_Gradients[i*n_features+f])>threshold)
        {
          diff=1;
          break;
        }
      if(diff==1)
        break;
    }
    if(diff==0)
    {
      break;
    }
    memcpy(V_Gradients,new_V_Gradients,n_states*n_features*sizeof(double));
  }


  // Q Gradient iteration
  int f;
  for(i=0;i<n_states;i++)
    for(j=0;j<n_actions;j++)
      for(k=0;k<n_states;k++)
        for(f=0;f<n_features;f++)
        {
          Q_Gradients[i*n_actions*n_features+j*n_features+f]+=transition_probabilities[i*n_states*n_actions+j*n_states+k]*(featureMatrix[k*n_features+f]+discount*V_Gradients[k*n_features+f]);
        }
}


long double gsoft_max(int n_actions, double approx_level, double*value_set)
{
  long  double sum_value=0;
  int i=0;
  for(i=0;i<n_actions;i++)
  {
    sum_value=sum_value+expl(value_set[i]*approx_level);
  }
  return logl(sum_value)/approx_level;
}
void gsoft_V_iteration(int n_states, int n_actions, double* transition_probabilities, double* reward, double discount, double threshold, double approx_level, double*V, double* new_V, double*value_update, double*Q)
{
  // Value iteration
  double diff=1;
  while(diff>threshold)
  {
    int i,j,k;
    for(i=0;i<n_states;i++)
    {
      for(j=0; j<n_actions;j++)
      {
        value_update[j]=0;
        for(k=0;k<n_states;k++)
        {
          value_update[j]+=transition_probabilities[i*n_states*n_actions+j*n_states+k]*(reward[k]+discount*V[k]);
        }
      }
      new_V[i]=gsoft_max(n_actions,approx_level,value_update);
    }
    diff=0;
    for(i=0;i<n_states;i++)
    {
      if(fabs(new_V[i]-V[i])>threshold)
      {
        diff=1;
        break;
      }
    }
    if(diff==0)
    {
      break;
    }
   memcpy(V,new_V,n_states*sizeof(double));
  }


  // Q value iteration
  int i,j,k;
  for(i=0;i<n_states;i++)
    for(j=0;j<n_actions;j++)
    {
      for(k=0;k<n_states;k++)
        Q[i*n_actions+j]=Q[i*n_actions+j]+transition_probabilities[i*n_states*n_actions+j*n_states+k]*(reward[k]+V[k]);
    }

}
void gsoft_G_iteration(int n_states, int n_actions, int n_features, double*featureMatrix, double* transition_probabilities, double* reward, double discount, double threshold, double approx_level, double*Values, double*QValues, double*V_Gradients, double*Q_Gradients,double*d_pnorm_vector,double*d_softmax_matrix,double*new_V_Gradients,double*d_sum_q)
{

  //V Gradient iteration
  double diff=1;
  // Precompute several important variables
  int i,j,k;
  for(i=0;i<n_states;i++)
  {
    long double sum_value=0;
    for(j=0;j<n_actions;j++)
    {
      sum_value=sum_value+expl(approx_level*QValues[i*n_actions+j]);
    }
    // d_softmax_matrix
    for(j=0;j<n_actions;j++)
    {
     d_softmax_matrix[i*n_actions+j]=expl(approx_level*QValues[i*n_actions+j])/sum_value;
    }
  }
  while(diff>threshold)
  {
    int i,j,k,f;
    for(i=0;i<n_states;i++)
    {
      for(f=0;f<n_features;f++)
        d_sum_q[f]=0;
      for(j=0; j<n_actions;j++)
      {
        double d_softmax=d_softmax_matrix[i*n_actions+j];
        for(k=0;k<n_states;k++)
        {
          // feature summation
          for(f=0;f<n_features;f++)
            d_sum_q[f]=d_sum_q[f]+d_softmax*transition_probabilities[i*n_states*n_actions+j*n_states+k]*(featureMatrix[k*n_features+f]+discount*V_Gradients[k*n_features+f]);
        }
      }
      for(f=0;f<n_features;f++)
        new_V_Gradients[i*n_features+f]=d_sum_q[f];
    }
    diff=0;
    for(i=0;i<n_states;i++)
    {
      for(f=0;f<n_features;f++)
        if(fabs(new_V_Gradients[i*n_features+f]-V_Gradients[i*n_features+f])>threshold)
        {
          diff=1;
          break;
        }
      if(diff==1)
        break;
    }
    if(diff==0)
    {
      break;
    }
    memcpy(V_Gradients,new_V_Gradients,n_states*n_features*sizeof(double));
  }


  // Q Gradient iteration
  int f;
  for(i=0;i<n_states;i++)
    for(j=0;j<n_actions;j++)
      for(k=0;k<n_states;k++)
        for(f=0;f<n_features;f++)
        {
          Q_Gradients[i*n_actions*n_features+j*n_features+f]+=transition_probabilities[i*n_states*n_actions+j*n_states+k]*(featureMatrix[k*n_features+f]+discount*V_Gradients[k*n_features+f]);
        }
}




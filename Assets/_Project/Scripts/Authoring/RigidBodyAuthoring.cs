using Latios.Psyshock;
using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace Authoring
{
    public class RigidBodyAuthoring : MonoBehaviour
    {
        private class RigidBodyBaker : Baker<RigidBodyAuthoring>
        {
            public override void Bake(RigidBodyAuthoring authoring)
            {
            }
        }
    }
    
    public struct RigidBody : IComponentData
    {
        public UnitySim.Velocity        velocity;
        public UnitySim.MotionExpansion motionExpansion;
        public RigidTransform           inertialPoseWorldTransform;
        public UnitySim.Mass            mass;
        public float                    coefficientOfFriction;
        public float                    coefficientOfRestitution;
        public float                    linearDamping;
        public float                    angularDamping;
    } 
}
using Authoring;
using Latios;
using Latios.Psyshock;
using Latios.Transforms;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

using static Unity.Entities.SystemAPI;

namespace System
{
    public partial struct PhysicsSystem : ISystem, ISystemNewScene
    {
        LatiosWorldUnmanaged latiosWorld;

        EntityQuery m_rigidBodyQuery;
        EntityQuery m_staticCollidersQuery;

        BuildCollisionLayerTypeHandles m_typeHandles;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            latiosWorld = state.GetLatiosWorldUnmanaged();

            m_rigidBodyQuery       = state.Fluent().With<RigidBody>(false).PatchQueryForBuildingCollisionLayer().Build();
            m_staticCollidersQuery = state.Fluent().Without<RigidBody>().PatchQueryForBuildingCollisionLayer().Build();

            m_typeHandles = new BuildCollisionLayerTypeHandles(ref state);
        }

        public void OnNewScene(ref SystemState state)
        {
            for (int i = 0; i < 100; i++)
            {
                float x = i % 5;
                float z = (i / 5) % 5;
                float y = (i / 25) + 0.1f * i + 5;
                
                CreateBoxColliderDynamicEntity(state.EntityManager, new float3(x, y, z));
            }
        }

        Entity CreateBoxColliderDynamicEntity(EntityManager entityManager, float3 position)
        {
            Collider collider = new BoxCollider(float3.zero, new float3(1f) * 0.5f);
            
            var entity = CreatePhysicsBodyEntity(entityManager, position, collider);
            entityManager.AddComponent<RigidBody>(entity);
            entityManager.SetComponentData(entity, new RigidBody()
            {
                mass = new UnitySim.Mass(){inverseMass = math.rcp(1f)},
                coefficientOfFriction    = 0.5f,
                coefficientOfRestitution = 0.5f,
                linearDamping            = 0.05f,
                angularDamping           = 0.05f,
            });

            return entity;
        }
        
        Entity CreatePhysicsBodyEntity(EntityManager entityManager, float3 position, Collider collider)
        {
            var entity = entityManager.CreateEntity(ComponentType.ReadWrite<WorldTransform>(), ComponentType.ReadWrite<Collider>());
            
            entityManager.SetComponentData(entity, new WorldTransform()
            {
                worldTransform = new TransformQvvs(position, quaternion.identity, 1f, new float3(1f))
            });
            
            entityManager.SetComponentData(entity, collider);

            return entity;
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var initialJh = state.Dependency;

            // Build static layer
            m_typeHandles.Update(ref state);
            var buildStaticLayerJh =
                Physics.BuildCollisionLayer(m_staticCollidersQuery, m_typeHandles).ScheduleParallel(out var staticLayer, state.WorldUpdateAllocator, initialJh);

            // Gravity, and build rigid body layer
            var rigidBodyCount         = m_rigidBodyQuery.CalculateEntityCountWithoutFiltering();
            var rigidBodyColliderArray = CollectionHelper.CreateNativeArray<ColliderBody>(rigidBodyCount, state.WorldUpdateAllocator, NativeArrayOptions.UninitializedMemory);
            var rigidBodyAabbArray     = CollectionHelper.CreateNativeArray<Aabb>(rigidBodyCount, state.WorldUpdateAllocator, NativeArrayOptions.UninitializedMemory);

            var buildRigidBodyJh = new BuildRigidBodiesJob
            {
                deltaTime     = Time.DeltaTime,
                colliderArray = rigidBodyColliderArray,
                aabbArray     = rigidBodyAabbArray
            }.ScheduleParallel(m_rigidBodyQuery, initialJh);
            buildRigidBodyJh = Physics.BuildCollisionLayer(rigidBodyColliderArray, rigidBodyAabbArray)
                               .ScheduleParallel(out var rigidBodyLayer, state.WorldUpdateAllocator, buildRigidBodyJh);

            // Find collisions, generate contacts, and generate contact constraints
            var pairStream            = new PairStream(rigidBodyLayer, state.WorldUpdateAllocator);
            var findBodyBodyProcessor = new FindBodyVsBodyProcessor
            {
                bodyLookup       = GetComponentLookup<RigidBody>(true),
                pairStream       = pairStream.AsParallelWriter(),
                deltaTime        = Time.DeltaTime,
                inverseDeltaTime = math.rcp(Time.DeltaTime)
            };
            var bodyVsBodyJh = Physics.FindPairs(in rigidBodyLayer, in findBodyBodyProcessor).ScheduleParallelUnsafe(buildRigidBodyJh);

            var findBodyEnvironmentProcessor = new FindBodyVsEnvironmentProcessor
            {
                bodyLookup       = GetComponentLookup<RigidBody>(true),
                pairStream       = pairStream.AsParallelWriter(),
                deltaTime        = Time.DeltaTime,
                inverseDeltaTime = math.rcp(Time.DeltaTime)
            };
            state.Dependency = Physics.FindPairs(in rigidBodyLayer, in staticLayer, in findBodyEnvironmentProcessor)
                .ScheduleParallelUnsafe(JobHandle.CombineDependencies(bodyVsBodyJh, buildStaticLayerJh));

            // Solve constraints
            int numIterations  = 8;
            var solveProcessor = new SolveBodiesProcessor
            {
                rigidBodyLookup        = GetComponentLookup<RigidBody>(false),
                invNumSolverIterations = math.rcp(numIterations)
            };
            for (int i = 0; i < numIterations; i++)
            {
                state.Dependency = Physics.ForEachPair(in pairStream, in solveProcessor).ScheduleParallel(state.Dependency);
            }

            // Integrate and update entities
            new IntegrateRigidBodiesJob { deltaTime = Time.DeltaTime }.ScheduleParallel();
            
            state.Dependency = new ColliderDrawingJob()
            {

            }.ScheduleParallel(state.Dependency);
        }

        struct ContactStreamData
        {
            public UnitySim.ContactJacobianBodyParameters                bodyParameters;
            public StreamSpan<UnitySim.ContactJacobianContactParameters> contactParameters;
            public StreamSpan<float>                                     contactImpulses;
        }

        [BurstCompile]
        partial struct BuildRigidBodiesJob : IJobEntity
        {
            public float deltaTime;

            [NativeDisableParallelForRestriction] public NativeArray<ColliderBody> colliderArray;
            [NativeDisableParallelForRestriction] public NativeArray<Aabb>         aabbArray;

            public void Execute(Entity entity, [EntityIndexInQuery] int index, ref RigidBody rigidBody, in Collider collider, in WorldTransform transform)
            {
                rigidBody.velocity.linear.y += -9.81f * deltaTime;

                var aabb                  = Physics.AabbFrom(in collider, in transform.worldTransform);
                var angularExpansion      = UnitySim.AngularExpansionFactorFrom(in collider);
                var motionExpansion       = new UnitySim.MotionExpansion(in rigidBody.velocity, deltaTime, angularExpansion);
                aabb                      = motionExpansion.ExpandAabb(aabb);
                rigidBody.motionExpansion = motionExpansion;

                colliderArray[index] = new ColliderBody
                {
                    collider  = collider,
                    transform = transform.worldTransform,
                    entity    = entity
                };
                aabbArray[index] = aabb;

                var localCenterOfMass = UnitySim.LocalCenterOfMassFrom(in collider);
                var localInertia      = UnitySim.LocalInertiaTensorFrom(in collider, transform.stretch);
                UnitySim.ConvertToWorldMassInertia(in transform.worldTransform,
                                                   in localInertia,
                                                   localCenterOfMass,
                                                   rigidBody.mass.inverseMass,
                                                   out rigidBody.mass,
                                                   out rigidBody.inertialPoseWorldTransform);
            }
        }

        struct FindBodyVsEnvironmentProcessor : IFindPairsProcessor
        {
            [ReadOnly] public ComponentLookup<RigidBody> bodyLookup;
            public PairStream.ParallelWriter             pairStream;
            public float                                 deltaTime;
            public float                                 inverseDeltaTime;

            DistanceBetweenAllCache distanceBetweenAllCache;

            public void Execute(in FindPairsResult result)
            {
                ref readonly var rigidBodyA = ref bodyLookup.GetRefRO(result.entityA).ValueRO;

                var maxDistance = UnitySim.MotionExpansion.GetMaxDistance(in rigidBodyA.motionExpansion);
                Physics.DistanceBetweenAll(result.colliderA, result.transformA, result.colliderB, result.transformB, maxDistance, ref distanceBetweenAllCache);
                foreach (var distanceResult in distanceBetweenAllCache)
                {
                    var contacts = UnitySim.ContactsBetween(result.colliderA, result.transformA, result.colliderB, result.transformB, in distanceResult);

                    ref var streamData           = ref pairStream.AddPairAndGetRef<ContactStreamData>(result.pairStreamKey, true, false, out var pair);
                    streamData.contactParameters = pair.Allocate<UnitySim.ContactJacobianContactParameters>(contacts.contactCount, NativeArrayOptions.UninitializedMemory);
                    streamData.contactImpulses   = pair.Allocate<float>(contacts.contactCount, NativeArrayOptions.ClearMemory);

                    UnitySim.BuildJacobian(streamData.contactParameters.AsSpan(),
                                           out streamData.bodyParameters,
                                           rigidBodyA.inertialPoseWorldTransform,
                                           in rigidBodyA.velocity,
                                           in rigidBodyA.mass,
                                           RigidTransform.identity,
                                           default,
                                           default,
                                           contacts.contactNormal,
                                           contacts.AsSpan(),
                                           rigidBodyA.coefficientOfRestitution,
                                           rigidBodyA.coefficientOfFriction,
                                           UnitySim.kMaxDepenetrationVelocityDynamicStatic,
                                           9.81f,
                                           deltaTime,
                                           inverseDeltaTime,
                                           1);
                }
            }
        }

        struct FindBodyVsBodyProcessor : IFindPairsProcessor
        {
            [ReadOnly] public ComponentLookup<RigidBody> bodyLookup;
            public PairStream.ParallelWriter             pairStream;
            public float                                 deltaTime;
            public float                                 inverseDeltaTime;

            DistanceBetweenAllCache distanceBetweenAllCache;

            public void Execute(in FindPairsResult result)
            {
                ref readonly var rigidBodyA = ref bodyLookup.GetRefRO(result.entityA).ValueRO;
                ref readonly var rigidBodyB = ref bodyLookup.GetRefRO(result.entityB).ValueRO;

                var maxDistance = UnitySim.MotionExpansion.GetMaxDistance(in rigidBodyA.motionExpansion, in rigidBodyB.motionExpansion);
                Physics.DistanceBetweenAll(result.colliderA, result.transformA, result.colliderB, result.transformB, maxDistance, ref distanceBetweenAllCache);
                foreach (var distanceResult in distanceBetweenAllCache)
                {
                    var contacts = UnitySim.ContactsBetween(result.colliderA, result.transformA, result.colliderB, result.transformB, in distanceResult);

                    var coefficientOfFriction    = math.sqrt(rigidBodyA.coefficientOfFriction * rigidBodyB.coefficientOfFriction);
                    var coefficientOfRestitution = math.sqrt(rigidBodyA.coefficientOfRestitution * rigidBodyB.coefficientOfRestitution);

                    ref var streamData           = ref pairStream.AddPairAndGetRef<ContactStreamData>(result.pairStreamKey, true, true, out var pair);
                    streamData.contactParameters = pair.Allocate<UnitySim.ContactJacobianContactParameters>(contacts.contactCount, NativeArrayOptions.UninitializedMemory);
                    streamData.contactImpulses   = pair.Allocate<float>(contacts.contactCount, NativeArrayOptions.ClearMemory);

                    UnitySim.BuildJacobian(streamData.contactParameters.AsSpan(),
                                           out streamData.bodyParameters,
                                           rigidBodyA.inertialPoseWorldTransform,
                                           in rigidBodyA.velocity,
                                           in rigidBodyA.mass,
                                           rigidBodyB.inertialPoseWorldTransform,
                                           in rigidBodyB.velocity,
                                           in rigidBodyB.mass,
                                           contacts.contactNormal,
                                           contacts.AsSpan(),
                                           coefficientOfRestitution,
                                           coefficientOfFriction,
                                           UnitySim.kMaxDepenetrationVelocityDynamicDynamic,
                                           9.81f,
                                           deltaTime,
                                           inverseDeltaTime,
                                           1);
                }
            }
        }

        struct SolveBodiesProcessor : IForEachPairProcessor
        {
            public PhysicsComponentLookup<RigidBody> rigidBodyLookup;
            public float                             invNumSolverIterations;

            public void Execute(ref PairStream.Pair pair)
            {
                ref var streamData = ref pair.GetRef<ContactStreamData>();

                ref var rigidBodyA = ref rigidBodyLookup.GetRW(pair.entityA).ValueRW;

                UnitySim.Velocity defaultVelocity = default;
                ref var           velocityB       = ref defaultVelocity;
                UnitySim.Mass     massB           = default;

                if (pair.bIsRW)
                {
                    ref var rigidBodyB = ref rigidBodyLookup.GetRW(pair.entityB).ValueRW;
                    velocityB          = ref rigidBodyB.velocity;
                    massB              = rigidBodyB.mass;
                }

                UnitySim.SolveJacobian(ref rigidBodyA.velocity,
                                       in rigidBodyA.mass,
                                       UnitySim.MotionStabilizer.kDefault,
                                       ref velocityB,
                                       in massB,
                                       UnitySim.MotionStabilizer.kDefault,
                                       streamData.contactParameters.AsSpan(),
                                       streamData.contactImpulses.AsSpan(),
                                       in streamData.bodyParameters,
                                       false,
                                       invNumSolverIterations,
                                       out _);
            }
        }

        [BurstCompile]
        partial struct IntegrateRigidBodiesJob : IJobEntity
        {
            public float deltaTime;

            public void Execute(TransformAspect transform, ref RigidBody rigidBody)
            {
                //Some velocity clamping stuff
                rigidBody.velocity.linear = math.clamp(math.length(rigidBody.velocity.linear), 0, 10) * math.normalizesafe(rigidBody.velocity.linear);
                
                var previousInertialPose = rigidBody.inertialPoseWorldTransform;
                UnitySim.Integrate(ref rigidBody.inertialPoseWorldTransform, ref rigidBody.velocity, rigidBody.linearDamping, rigidBody.angularDamping, deltaTime);
                transform.worldTransform = UnitySim.ApplyInertialPoseWorldTransformDeltaToWorldTransform(transform.worldTransform,
                                                                                                         in previousInertialPose,
                                                                                                         in rigidBody.inertialPoseWorldTransform);
            }
        }
        
        [BurstCompile]
        partial struct ColliderDrawingJob : IJobEntity
        {
            void Execute(in WorldTransform worldTransform,
                in Collider collider)
            {
                var rigidTransform = new RigidTransform(worldTransform.worldTransform.ToMatrix4x4());
                if (collider.type == ColliderType.TriMesh)
                {
                    var triMeshCollider = (TriMeshCollider)collider;
                
                    PhysicsDebug.DrawCollider(in triMeshCollider, in rigidTransform, UnityEngine.Color.red);
                }
                else if(collider.type == ColliderType.Box)
                {
                    var boxCollider = (BoxCollider)collider;
                    PhysicsDebug.DrawCollider(in boxCollider, in rigidTransform, UnityEngine.Color.red);
                }
            }
        }
    }
}